/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <drm_fourcc.h>
#include <linux/kd.h>
#include <linux/vt.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>


#include "NvUtils.h"
#include "video_dec_drm.h"
#include "tegra_drm.h"
#ifndef DOWNSTREAM_TEGRA_DRM
#include "tegra_drm_nvdc.h"
#endif
#include "NvApplicationProfiler.h"

#define TEST_ERROR(cond, str, label) \
    if(cond) \
    { \
        cerr << str << endl; \
        error = 1; \
        goto label; \
    }

#define CHUNK_SIZE 4000000
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define INVALID_PLANE 0xFFFF
#define ZERO_FD 0x0

using namespace std;

extern unsigned int image_w;
extern unsigned int image_h;
extern char image_pixels_array[];

unordered_map <int, int> fd_map;

static void leave_vt(context_t * ctx)
{
    int ret;

    ret = ioctl(ctx->console_fd, KDSETMODE, KD_TEXT);
    if (ret < 0) {
        printf("KDSETMODE failed, err=%s\n", strerror(errno));
    }

    if (ctx->active_vt >= 0) {
        ret = ioctl(ctx->console_fd, VT_ACTIVATE, ctx->active_vt);
        if (ret < 0) {
            printf("VT_ACTIVATE failed, err=%s\n", strerror(errno));
        }

        ret = ioctl(ctx->console_fd, VT_WAITACTIVE, ctx->active_vt);
        if (ret < 0) {
            printf("VT_WAITACTIVE failed, err= %s\n", strerror(errno));
        }
    }

    close(ctx->console_fd);
    ctx->console_fd = -1;
    ctx->active_vt = -1;
}

static void enter_vt(context_t * ctx)
{
    int i, ret, fd, vtno;
    struct vt_stat vts;
    const char *vcs[] = { "/dev/vc/%d", "/dev/tty%d", NULL };
    static char vtname[11];

    fd = open("/dev/tty0", O_WRONLY, 0);
    if (fd < 0) {
        printf("can't open /dev/tty0 err=%s\n", strerror(errno));
        return;
    }

    ret = ioctl(fd, VT_OPENQRY, &vtno);
    if (ret < 0) {
        printf("VT_OPENQRY failed, err=%s\n", strerror(errno));
        close(fd);
        return;
    }

    if (vtno == -1) {
        printf("can't find free VT\n");
        close(fd);
        return;
    }

    printf("Using VT number %d\n", vtno);
    close(fd);

    i = 0;
    while (vcs[i] != NULL) {
        snprintf(vtname, sizeof(vtname), vcs[i], vtno);
        ctx->console_fd = open(vtname, O_RDWR | O_NDELAY, 0);
        if (ctx->console_fd >= 0) {
            break;
        }
        i++;
    }

    if (ctx->console_fd < 0) {
        printf("can't open virtual console %d\n", vtno);
    }

    ret = ioctl(ctx->console_fd, VT_GETSTATE, &vts);
    if (ret < 0) {
        printf("VT_GETSTATE failed, err=%s\n", strerror(errno));
    } else {
        ctx->active_vt = vts.v_active;
    }

    ret = ioctl(ctx->console_fd, VT_ACTIVATE, vtno);
    if (ret < 0) {
        printf("VT_ACTIVATE failed, err=%s\n", strerror(errno));
        return;
    }

    ret = ioctl(ctx->console_fd, VT_WAITACTIVE, vtno);
    if (ret < 0) {
        printf("VT_WAITACTIVE failed, err=%s\n", strerror(errno));
        return;
    }

    ret = ioctl(ctx->console_fd, KDSETMODE, KD_GRAPHICS);
    if (ret < 0) {
        printf("KDSETMODE KD_GRAPHICS failed, err=%s\n", strerror(errno));
    }

    return;
}

static int
read_decoder_input_chunk(ifstream * stream, NvBuffer * buffer)
{
    // Length is the size of the buffer in bytes
    streamsize bytes_to_read = MIN(CHUNK_SIZE, buffer->planes[0].length);

    stream->read((char *) buffer->planes[0].data, bytes_to_read);
    // It is necessary to set bytesused properly, so that decoder knows how
    // many bytes in the buffer are valid
    buffer->planes[0].bytesused = stream->gcount();
    return 0;
}

static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    ctx->dec->abort();
    leave_vt(ctx);
    if (ctx->conv)
    {
        ctx->conv->abort();
        pthread_cond_broadcast(&ctx->queue_cond);
    }
}

static bool
conv0_output_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf,
                                   NvBuffer * buffer, NvBuffer * shared_buffer,
                                   void *arg)
{
    context_t *ctx = (context_t *) arg;
    struct v4l2_buffer dec_capture_ret_buffer;
    struct v4l2_plane planes[MAX_PLANES];

    if (!v4l2_buf)
    {
        cerr << "Error while dequeueing conv output plane buffer" << endl;
        abort(ctx);
        return false;
    }

    if (v4l2_buf->m.planes[0].bytesused == 0)
    {
        return false;
    }

    memset(&dec_capture_ret_buffer, 0, sizeof(dec_capture_ret_buffer));
    memset(planes, 0, sizeof(planes));

    dec_capture_ret_buffer.index = shared_buffer->index;
    dec_capture_ret_buffer.m.planes = planes;

    pthread_mutex_lock(&ctx->queue_lock);
    ctx->conv_output_plane_buf_queue->push(buffer);

    // Return the buffer dequeued from converter output plane
    // back to decoder capture plane
    if (ctx->dec->capture_plane.qBuffer(dec_capture_ret_buffer, NULL) < 0)
    {
        abort(ctx);
        pthread_mutex_unlock(&ctx->queue_lock);
        return false;
    }

    pthread_cond_broadcast(&ctx->queue_cond);
    pthread_mutex_unlock(&ctx->queue_lock);

    return true;
}

static bool
conv0_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf,
                                    NvBuffer * buffer, NvBuffer * shared_buffer,
                                    void *arg)
{
    context_t *ctx = (context_t *) arg;

    if (!v4l2_buf)
    {
        cerr << "Error while dequeueing conv capture plane buffer" << endl;
        abort(ctx);
        return false;
    }

    if (v4l2_buf->m.planes[0].bytesused == 0)
    {
        // Enqueue an invalid fd to send EOS signal to drm rendering thread
        ctx->drm_renderer->enqueBuffer(-1);
        return false;
    }

    // Enqueue the fd into drm rendering queue
    // fd_map is used to store the context of v4l2 buff
    fd_map.insert(make_pair(buffer->planes[0].fd, v4l2_buf->index));
    ctx->drm_renderer->enqueBuffer(buffer->planes[0].fd);

    // To fix the flicker issue(Bug 200292247), we seperate the
    // the buffer enqueue process from this callback into
    // the thread renderer_dequeue_loop and make it running
    // asynchronously.

    return true;
}

static int
sendEOStoConverter(context_t *ctx)
{
    // Check if converter is running
    if (ctx->conv->output_plane.getStreamStatus())
    {
        NvBuffer *conv_buffer;
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(&planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        pthread_mutex_lock(&ctx->queue_lock);
        while (ctx->conv_output_plane_buf_queue->empty())
        {
            pthread_cond_wait(&ctx->queue_cond, &ctx->queue_lock);
        }
        conv_buffer = ctx->conv_output_plane_buf_queue->front();
        ctx->conv_output_plane_buf_queue->pop();
        pthread_mutex_unlock(&ctx->queue_lock);

        v4l2_buf.index = conv_buffer->index;

        // Queue EOS buffer on converter output plane
        return ctx->conv->output_plane.qBuffer(v4l2_buf, NULL);
    }

    return 0;
}

static void *
ui_render_loop_fcn(void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvDrmFB ui_fb[3];
    uint32_t plane_count = ctx->drm_renderer->getPlaneCount();
    uint32_t plane_index = 0;
    // The variables 'image_pixels_array', 'image_w' and 'image_h'
    // are defined in the following auto-generated header file,
    // 'image_rgba.h'
    const char *p = image_pixels_array;
    uint32_t ui_width = 200;
    uint32_t ui_height = 200;
    uint32_t frame = 0;
    long elapsed_us = 0;

    // Render a static JPEG image on the first plane
    ctx->drm_renderer->createDumbFB(image_w, image_h,
            DRM_FORMAT_ARGB8888,
            &ui_fb[0]);
    for (uint32_t y = 0; y < image_h; ++y)
    {
        for (uint32_t x = 0; x < image_w; ++x)
        {
            uint32_t off = ui_fb[0].bo[0].pitch * y + x * 4;
            ui_fb[0].bo[0].data[off] = *p++;
            ui_fb[0].bo[0].data[off + 1] = *p++;
            ui_fb[0].bo[0].data[off + 2] = *p++;
            ui_fb[0].bo[0].data[off + 3] = *p++;
        }
    }

    // It's kind of trick to distinguish the platforms with plane_count.
    // We'd better decide target window with the real hardware configuration.
    // By default,
    // TX1:
    //    CRTC 0: primary(win_A), overlay planes(win_B & win_C & win_D)
    //    CRTC 1: primary(win_A), overlay planes(win_B & win_C)
    // TX2:
    //    CRTC 0: primary(win_A), overlay planes(win_B & win_C)
    //    CRTC 1: primary(win_A), overlay planes(win_B)
    //    CRTC 2: primary(win_A)
    // NOTE: The plane_count implies the overlay windows

    if (plane_count == 3)
        plane_index = (ctx->crtc == 0) ? 0 : 2;
    else
        plane_index = (ctx->crtc == 0) ? 0 : 3;

    ctx->drm_renderer->setPlane(plane_index, ui_fb[0].fb_id,
            0, 0, image_w, image_h,
            0, 0, image_w << 16, image_h << 16);

    // Moving color block on the second plane
    // The ui_fb[1] and ui_fb[2] are playing the roles of
    // double buffering to get rid of tearing issue.
    for (uint32_t i = 1; i < 3; i++)
        ctx->drm_renderer->createDumbFB(ui_height, ui_width,
                DRM_FORMAT_ARGB8888,
                &ui_fb[i]);
    do {
        struct timeval begin, end;

        gettimeofday(&begin, NULL);

        for (uint32_t y = 0; y < ui_height; ++y)
        {
            for (uint32_t x = 0; x < ui_width; ++x)
            {
                uint32_t off = ui_fb[frame % 2 + 1].bo[0].pitch * y + x * 4;
                ui_fb[frame % 2 + 1].bo[0].data[off] = frame % 255;
                ui_fb[frame % 2 + 1].bo[0].data[off + 1] = (frame + 255 / 3) % 255;
                ui_fb[frame % 2 + 1].bo[0].data[off + 2] = (frame + 255 / 2)% 255;
                ui_fb[frame % 2 + 1].bo[0].data[off + 3] = x % 255;
            }
        }

        // If plane_count is equal to 3, we don't have enough overlay to render
        // moving color block on the second crtc.
        if (plane_count == 3)
            plane_index = (ctx->crtc == 0) ? 1 : INVALID_PLANE;
        else
            plane_index = (ctx->crtc == 0) ? 1 : 4;

        // The flip will be happening after vblank for the completed buffer
        ctx->drm_renderer->setPlane(plane_index, ui_fb[frame % 2 + 1].fb_id,
                frame % image_w, frame % image_h, ui_width, ui_height,
                0, 0, ui_width << 16, ui_height << 16);

        frame++;

        // Get EOS signal from video capturing thread,
        // so setPlane(fd=0) to disable the windows before exiting
        if (ctx->got_eos && ctx->got_exit)
        {
            if (plane_count == 3)
                plane_index = (ctx->crtc == 0) ? 0 : 2;
            else
                plane_index = (ctx->crtc == 0) ? 0 : 3;
            ctx->drm_renderer->setPlane(plane_index, ZERO_FD,
                    0, 0, image_w, image_h,
                    0, 0, image_w << 16, image_h << 16);

            if (plane_count == 3)
                plane_index = (ctx->crtc == 0) ? 1 : INVALID_PLANE;
            else
                plane_index = (ctx->crtc == 0) ? 1 : 4;
            ctx->drm_renderer->setPlane(plane_index, ZERO_FD,
                    0, 0, image_w, image_h,
                    0, 0, image_w << 16, image_h << 16);

            break;
        }

        gettimeofday(&end, NULL);
        elapsed_us = (end.tv_sec - begin.tv_sec) * 1000000 +
            (end.tv_usec - begin.tv_usec);
        if (elapsed_us < (1000000 / ctx->fps))
            usleep((1000000 / ctx->fps) - elapsed_us);
    } while (!ctx->got_error);

    // Destroy the dumb framebuffers
    for (uint32_t i = 0; i < 3; i++)
        ctx->drm_renderer->removeFB(ui_fb[i].fb_id);

    return NULL;
}

static void *
renderer_dequeue_loop_fcn(void *args)
{
    context_t *ctx = (context_t *) args;
    int fd;
    int index;

    while (true)
    {
        fd = ctx->drm_renderer->dequeBuffer();

        // Received EOS signal from drm rendering thread
        if (fd == -1)
            return NULL;

        // Return the buffer back into converter capture plane in time
        // to avoid starving it
        auto map_entry = fd_map.find(fd);
        if (map_entry != fd_map.end())
        {
            index = (int) map_entry->second;
            fd_map.erase(fd);
            struct v4l2_buffer buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset (&buf, 0 , sizeof(buf));
            memset(planes, 0, sizeof(planes));
            buf.index = index;
            buf.m.planes = planes;
            buf.m.planes[0].m.fd = fd;

            if (ctx->conv->capture_plane.qBuffer(buf, NULL) < 0)
            {
                cerr << "Error while enqueueing conv capture plane buffer" << endl;
                abort(ctx);
                return NULL;
            }
        }
        else
        {
            cerr << "Error while retrieving the fd from map list" << endl;
            abort(ctx);
            return NULL;
        }
    };

    return NULL;
}

static void
query_and_set_capture(context_t * ctx)
{
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    int32_t min_dec_capture_buffers;
    int ret = 0;
    int error = 0;
    uint32_t window_width;
    uint32_t window_height;
    v4l2_ctrl_video_displaydata displaydata;
    v4l2_ctrl_video_hdrmasteringdisplaydata v4l2_hdrmetadata;
    struct drm_tegra_hdr_metadata_smpte_2086 drm_metadata;

    memset(&v4l2_hdrmetadata,0,sizeof(v4l2_ctrl_video_hdrmasteringdisplaydata));
    memset(&drm_metadata,0,sizeof(struct drm_tegra_hdr_metadata_smpte_2086));

    // Get capture plane format from the decoder. This may change after
    // an resolution change event
    ret = dec->capture_plane.getFormat(format);
    TEST_ERROR(ret < 0,
               "Error: Could not get format from decoder capture plane", error);

    // Get the display resolution from the decoder
    ret = dec->capture_plane.getCrop(crop);
    TEST_ERROR(ret < 0,
               "Error: Could not get crop from decoder capture plane", error);

    cout << "Video Resolution: " << crop.c.width << "x" << crop.c.height
        << endl;

    ret = dec->checkifMasteringDisplayDataPresent(displaydata);
    if (ret == 0)
    {
        if (displaydata.masteringdisplaydatapresent)
        {
            ctx->streamHDR = true;
            ret = dec->MasteringDisplayData(&v4l2_hdrmetadata);
            TEST_ERROR(ret < 0,
                    "Error while getting HDR mastering display data",
                    error);
                memcpy(&drm_metadata,&v4l2_hdrmetadata,sizeof(v4l2_ctrl_video_hdrmasteringdisplaydata));
        }
        else
            cout << "APP_INFO : mastering display data not found" << endl;
    }

    // For file write, first deinitialize output and capture planes
    // of video converter and then use the new resolution from
    // decoder event resolution change
    if (ctx->conv)
    {
        ret = sendEOStoConverter(ctx);
        TEST_ERROR(ret < 0,
                   "Error while queueing EOS buffer on converter output",
                   error);

        ctx->conv->capture_plane.waitForDQThread(2000);

        ctx->conv->output_plane.deinitPlane();
        ctx->conv->capture_plane.deinitPlane();

        while(!ctx->conv_output_plane_buf_queue->empty())
        {
            ctx->conv_output_plane_buf_queue->pop();
        }
    }

    // Destroy the old instance of renderer as resolution might have changed
    if (ctx->drm_renderer)
        delete ctx->drm_renderer;

    if (ctx->window_width && ctx->window_height)
    {
        // As specified by user on commandline
        window_width = ctx->window_width;
        window_height = ctx->window_height;
    }
    else
    {
        // If we render both UI and video stream, here it scales down
        // video stream by 2 to get a better user experience
        if (!ctx->disable_ui && !ctx->disable_video)
        {
            window_width =  crop.c.width / 2;
            window_height =  crop.c.height / 2;
        }
        else
        {
            // Resolution got from the decoder
            window_width = crop.c.width;
            window_height = crop.c.height;
        }
    }

    ctx->drm_renderer = NvDrmRenderer::createDrmRenderer("renderer0",
            window_width, window_height, ctx->window_x, ctx->window_y,
             ctx->connector, ctx->crtc, drm_metadata, ctx->streamHDR);

    TEST_ERROR(!ctx->drm_renderer,
            "Error in setting up drm renderer", error);

    ctx->drm_renderer->setFPS(ctx->fps);

    // Enable data profiling for renderer
    if (ctx->stats)
        ctx->drm_renderer->enableProfiling();

    if (!ctx->disable_ui)
    {
        pthread_create(&ctx->ui_renderer_loop, NULL,
                ui_render_loop_fcn, ctx);
    }

    // deinitPlane unmaps the buffers and calls REQBUFS with count 0
    dec->capture_plane.deinitPlane();

    // Not necessary to call VIDIOC_S_FMT on decoder capture plane.
    // But decoder setCapturePlaneFormat function updates the class variables
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

    // Get the minimum buffers which have to be requested on the capture plane
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    TEST_ERROR(ret < 0,
               "Error while getting value of minimum capture plane buffers",
               error);

    // Request (min + 5) buffers, export and map buffers
    ret =
        dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                       min_dec_capture_buffers + 5, false,
                                       false);
    TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);

    if (ctx->conv)
    {
        ret = ctx->conv->setOutputPlaneFormat(format.fmt.pix_mp.pixelformat,
                                              format.fmt.pix_mp.width,
                                              format.fmt.pix_mp.height,
                                              V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR);
        TEST_ERROR(ret < 0, "Error in converter output plane set format",
                   error);

        ret = ctx->conv->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                                crop.c.width,
                                                crop.c.height,
                                                V4L2_NV_BUFFER_LAYOUT_PITCH);
        TEST_ERROR(ret < 0, "Error in converter capture plane set format",
                   error);

        ret = ctx->conv->setCropRect(0, 0, crop.c.width, crop.c.height);
        TEST_ERROR(ret < 0, "Error while setting crop rect", error);

        ret =
            ctx->conv->output_plane.setupPlane(V4L2_MEMORY_DMABUF,
                                                dec->capture_plane.
                                                getNumBuffers(), false, false);
        TEST_ERROR(ret < 0, "Error in converter output plane setup", error);

        ret =
            ctx->conv->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                                 dec->capture_plane.
                                                 getNumBuffers(), true, false);
        TEST_ERROR(ret < 0, "Error in converter capture plane setup", error);

        ret = ctx->conv->output_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in converter output plane streamon", error);

        ret = ctx->conv->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in converter output plane streamoff", error);

        // Add all empty conv output plane buffers to conv_output_plane_buf_queue
        for (uint32_t i = 0; i < ctx->conv->output_plane.getNumBuffers(); i++)
        {
            ctx->conv_output_plane_buf_queue->push(ctx->conv->output_plane.
                    getNthBuffer(i));
        }

        for (uint32_t i = 0; i < ctx->conv->capture_plane.getNumBuffers(); i++)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));

            v4l2_buf.index = i;
            v4l2_buf.m.planes = planes;
            ret = ctx->conv->capture_plane.qBuffer(v4l2_buf, NULL);
            TEST_ERROR(ret < 0, "Error Qing buffer at converter capture plane",
                       error);
        }
        ctx->conv->output_plane.startDQThread(ctx);
        ctx->conv->capture_plane.startDQThread(ctx);

    }

    // Capture plane STREAMON
    ret = dec->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

    // Dequeue buffers from render and return it back into converter
    pthread_create(&ctx->renderer_dequeue_loop, NULL, renderer_dequeue_loop_fcn, ctx);

    // Enqueue all the empty capture plane buffers
    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        TEST_ERROR(ret < 0, "Error Qing buffer at output plane", error);
    }
    cout << "Query and set capture successful" << endl;
    return;

error:
    if (error)
    {
        abort(ctx);
        cerr << "Error in " << __func__ << endl;
    }
}

static void *
dec_capture_loop_fcn(void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_event ev;
    int ret;

    cout << "Starting decoder capture loop thread" << endl;

    // Need to wait for the first Resolution change event, so that
    // the decoder knows the stream resolution and can allocate appropriate
    // buffers when we call REQBUFS
    do
    {
        ret = dec->dqEvent(ev, 50000);
        if (ret < 0)
        {
            if (errno == EAGAIN)
            {
                cerr <<
                    "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE"
                    << endl;
            }
            else
            {
                cerr << "Error in dequeueing decoder event" << endl;
            }
            abort(ctx);
            break;
        }
    }
    while (ev.type != V4L2_EVENT_RESOLUTION_CHANGE);

    // query_and_set_capture acts on the resolution change event
    if (!ctx->got_error)
        query_and_set_capture(ctx);

    // Exit on error or EOS which is signalled in main()
    while (!(ctx->got_error || dec->isInError() || ctx->got_eos))
    {
        NvBuffer *dec_buffer;

        // Check for Resolution change again
        ret = dec->dqEvent(ev, false);
        if (ret == 0)
        {
            switch (ev.type)
            {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    query_and_set_capture(ctx);
                    continue;
            }
        }

        while (1)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
            NvBuffer *conv_buffer;
            struct v4l2_buffer conv_output_buffer;
            struct v4l2_plane conv_planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            // Dequeue a filled buffer
            if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0))
            {
                if (errno == EAGAIN)
                {
                    usleep(1000);
                }
                else
                {
                    abort(ctx);
                    cerr << "Error while calling dequeue at capture plane" <<
                        endl;
                }
                break;
            }

            // Give the buffer to video converter output plane
            // instead of returning the buffer back to decoder capture plane
            memset(&conv_output_buffer, 0, sizeof(conv_output_buffer));
            memset(conv_planes, 0, sizeof(conv_planes));
            conv_output_buffer.m.planes = conv_planes;

            // Get an empty conv output plane buffer from conv_output_plane_buf_queue
            pthread_mutex_lock(&ctx->queue_lock);
            while (ctx->conv_output_plane_buf_queue->empty())
            {
                pthread_cond_wait(&ctx->queue_cond, &ctx->queue_lock);
            }
            conv_buffer = ctx->conv_output_plane_buf_queue->front();
            ctx->conv_output_plane_buf_queue->pop();
            pthread_mutex_unlock(&ctx->queue_lock);

            conv_output_buffer.index = conv_buffer->index;

            if (ctx->conv->output_plane.
                    qBuffer(conv_output_buffer, dec_buffer) < 0)
            {
                abort(ctx);
                cerr <<
                    "Error while queueing buffer at converter output plane"
                    << endl;
                break;
            }
        }
    }
    // Send EOS to converter
    if (ctx->conv)
    {
        if (sendEOStoConverter(ctx) < 0)
        {
            cerr << "Error while queueing EOS buffer on converter output"
                 << endl;
        }
    }
    cout << "Exiting decoder capture loop thread" << endl;
    return NULL;
}

static void
set_defaults(context_t * ctx)
{
    ctx->dec = NULL;
    ctx->decoder_pixfmt = 1;
    ctx->in_file_path = NULL;
    ctx->in_file = NULL;

    ctx->conv = NULL;

    ctx->drm_renderer = NULL;
    ctx->disable_video = false;
    ctx->disable_ui = false;
    ctx->console_fd = -1;
    ctx->active_vt = -1;
    ctx->crtc = 0;
    ctx->connector = 0;
    ctx->window_height = 0;
    ctx->window_width = 0;
    ctx->window_x = 0;
    ctx->window_y = 0;
    ctx->fps = 30;
    ctx->renderer_dequeue_loop = 0;

    ctx->conv_output_plane_buf_queue = new queue < NvBuffer * >;
    pthread_mutex_init(&ctx->queue_lock, NULL);
    pthread_cond_init(&ctx->queue_cond, NULL);

    ctx->dec_capture_loop = 0;
    ctx->got_error = false;
    ctx->got_eos = false;

    ctx->ui_renderer_loop = 0;
    ctx->got_exit = false;
    ctx->streamHDR = false;

    ctx->stress_iteration = 0;
    ctx->stats = false;
}

static resolution res_array[] = {
    {1920, 1080},
    {1280, 720},
    {960, 640},
    {640, 480},
};

static int
drm_rendering(context_t &ctx, int argc, char *argv[], int iteration)
{
    int ret = 0;
    int error = 0;
    uint32_t i;
    bool eos = false;
    NvApplicationProfiler &profiler = NvApplicationProfiler::getProfilerInstance();
    struct drm_tegra_hdr_metadata_smpte_2086 metadata;

    set_defaults(&ctx);

    enter_vt(&ctx);

    if (parse_csv_args(&ctx, argc, argv))
    {
        fprintf(stderr, "Error parsing commandline arguments\n");
        return -1;
    }

    if (ctx.stress_iteration)
    {
        cout << "\nStart the iteration: " << iteration << "\n" <<endl;
        ctx.window_width = res_array[iteration % 4].width;
        ctx.window_height = res_array[iteration % 4].height;
    }

    if (ctx.stats)
        profiler.start(NvApplicationProfiler::DefaultSamplingInterval);

    // Render UI infinitely until user terminate it
    if (ctx.disable_video)
    {
        ctx.drm_renderer = NvDrmRenderer::createDrmRenderer("renderer0",
                image_w, image_h, 0, 0, ctx.connector, ctx.crtc, metadata, ctx.streamHDR);

        TEST_ERROR(!ctx.drm_renderer, "Error creating drm renderer", cleanup);

        ctx.drm_renderer->setFPS(ctx.fps);

        // Enable data profiling for renderer
        if (ctx.stats)
            ctx.drm_renderer->enableProfiling();

        pthread_create(&ctx.ui_renderer_loop, NULL,
                ui_render_loop_fcn, &ctx);

        goto cleanup;
    }
    // Otherwise, it renders both video and UI, or render only video
    // when the option '--disable-ui' specified

    // The pipelie of this case is,
    // File --> Decoder --> Converter(YUV422/BL -> NV12/PL) --> DRM

    // ** Step 1 - Create video decoder **
    ctx.dec = NvVideoDecoder::createVideoDecoder("dec0");
    TEST_ERROR(!ctx.dec, "Could not create decoder", cleanup);

    // Enable data profiling for decoder
    if (ctx.stats)
        ctx.dec->enableProfiling();

    // Subscribe to Resolution change event
    ret = ctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE",
               cleanup);

    // Set format on the decoder output plane
    ret = ctx.dec->setOutputPlaneFormat(ctx.decoder_pixfmt, CHUNK_SIZE);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    // Set V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control to false
    // so that application can send chunks of encoded data instead of forming
    // complete frames.
    ret = ctx.dec->setFrameInputMode(1);
    TEST_ERROR(ret < 0,
            "Error in decoder setFrameInputMode", cleanup);

    // Query, Export and Map the output plane buffers so that we can read
    // encoded data into the buffers
    ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Error while setting up output plane", cleanup);

    ctx.in_file = new ifstream(ctx.in_file_path);
    TEST_ERROR(!ctx.in_file->is_open(), "Error opening input file", cleanup);

    // ** Step 2 - Create video converter **
    // Do conversion from YUV422/BL to NV12/PL
    ctx.conv = NvVideoConverter::createVideoConverter("conv0");
    TEST_ERROR(!ctx.conv, "Could not create video converter", cleanup);
    ctx.conv->output_plane.
        setDQThreadCallback(conv0_output_dqbuf_thread_callback);
    ctx.conv->capture_plane.
        setDQThreadCallback(conv0_capture_dqbuf_thread_callback);

    // Enable data profiling for converter
    if (ctx.stats)
        ctx.conv->enableProfiling();

    ret = ctx.dec->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane stream on", cleanup);

    // ** Step 3 - Set up decoder and converter in sub-thread **
    pthread_create(&ctx.dec_capture_loop, NULL, dec_capture_loop_fcn, &ctx);

    // ** Step 4 - feed the encoded data into decoder output plane**
    // Read encoded data and enqueue all the output plane buffers.
    // Exit loop in case file read is complete.
    i = 0;
    while (!eos && !ctx.got_error && !ctx.dec->isInError() &&
           i < ctx.dec->output_plane.getNumBuffers())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        buffer = ctx.dec->output_plane.getNthBuffer(i);

        read_decoder_input_chunk(ctx.in_file, buffer);

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        // It is necessary to queue an empty buffer to signal EOS to the decoder
        // i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
        i++;
    }

    // Since all the output plane buffers have been queued, we first need to
    // dequeue a buffer from output plane before we can read new data into it
    // and queue it again.
    while (!eos && !ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
        if (ret < 0)
        {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }

        read_decoder_input_chunk(ctx.in_file, buffer);

        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
    }

    // After sending EOS, all the buffers from output plane should be dequeued.
    // and after that capture plane loop should be signalled to stop.
    while (ctx.dec->output_plane.getNumQueuedBuffers() > 0 &&
           !ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if (ret < 0)
        {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
    }

    // Signal EOS to the decoder capture loop
    ctx.got_eos = true;

    if (ctx.conv)
    {
        ctx.conv->capture_plane.waitForDQThread(-1);
    }

    if (ctx.stats)
    {
        profiler.stop();
        if (ctx.dec)
            ctx.dec->printProfilingStats(cout);
        if (ctx.conv)
            ctx.conv->printProfilingStats(cout);
        if (ctx.drm_renderer)
            ctx.drm_renderer->printProfilingStats(cout);
        profiler.printProfilerData(cout);
    }

cleanup:
    if (ctx.dec_capture_loop)
    {
        pthread_join(ctx.dec_capture_loop, NULL);
    }

    if (ctx.renderer_dequeue_loop)
    {
        pthread_join(ctx.renderer_dequeue_loop, NULL);
    }

    if (ctx.ui_renderer_loop)
    {
        ctx.got_exit = true;
        pthread_join(ctx.ui_renderer_loop, NULL);
    }

    if (ctx.conv && ctx.conv->isInError())
    {
        cerr << "Converter is in error" << endl;
        error = 1;
    }

    if (ctx.dec && ctx.dec->isInError())
    {
        cerr << "Decoder is in error" << endl;
        error = 1;
    }

    if (ctx.got_error)
    {
        error = 1;
    }

    leave_vt(&ctx);

    // The decoder destructor does all the cleanup i.e set streamoff on output and capture planes,
    // unmap buffers, tell decoder to deallocate buffer (reqbufs ioctl with counnt = 0),
    // and finally call v4l2_close on the fd.
    delete ctx.dec;
    delete ctx.conv;

    // Similarly, NvDrmRenderer destructor does all the cleanup
    delete ctx.drm_renderer;
    delete ctx.in_file;
    delete ctx.conv_output_plane_buf_queue;

    free(ctx.in_file_path);

    if (ctx.stress_iteration)
    {
        if (error)
            cout << "\nERROR: failed in iteration: " << iteration << endl;
        else
            cout << "\nEnd the iteration: " << iteration << "\n" <<endl;
    }

    return error;
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int ret = 0;
    uint32_t iteration = 1;

    do
    {
        // Main loop to render UI(ARGB) or Video(YUV420) stream
        ret = drm_rendering(ctx, argc, argv, iteration);

    } while (iteration++ < ctx.stress_iteration && ret == 0);

    if (ret)
    {
        cout << "App run failed" << endl;
    }
    else
    {
        cout << "App run was successful" << endl;
    }

    return -ret;
}
