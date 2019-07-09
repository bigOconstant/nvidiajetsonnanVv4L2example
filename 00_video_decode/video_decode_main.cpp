/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "NvApplicationProfiler.h"
#include "NvUtils.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <nvbuf_utils.h>

#include "video_decode.h"
#include "nvbuf_utils.h"

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define MICROSECOND_UNIT 1000000
#define CHUNK_SIZE 4000000
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        !buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
        (buffer_ptr[2] == 1))

#define H264_NAL_UNIT_CODED_SLICE  1
#define H264_NAL_UNIT_CODED_SLICE_IDR  5

#define HEVC_NUT_TRAIL_N  0
#define HEVC_NUT_RASL_R  9
#define HEVC_NUT_BLA_W_LP  16
#define HEVC_NUT_CRA_NUT  21

#define IVF_FILE_HDR_SIZE   32
#define IVF_FRAME_HDR_SIZE  12

#define IS_H264_NAL_CODED_SLICE(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE)
#define IS_H264_NAL_CODED_SLICE_IDR(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE_IDR)

#define GET_H265_NAL_UNIT_TYPE(buffer_ptr) ((buffer_ptr[0] & 0x7E) >> 1)

#define IS_SEMIPLANAR_FMT(pixel_format) ((pixel_format == NvBufferColorFormat_NV12) || \
        (pixel_format == NvBufferColorFormat_NV12_ER) || \
        (pixel_format == NvBufferColorFormat_NV12_709) || \
        (pixel_format == NvBufferColorFormat_NV12_709_ER) || \
        (pixel_format == NvBufferColorFormat_NV12_2020))

using namespace std;

static int
dump_dmabuf(int dmabuf_fd,
                unsigned int plane,
                std::ofstream * stream)
{
    if (dmabuf_fd <= 0)
        return -1;

    int ret = -1;
    NvBufferParams parm;
    ret = NvBufferGetParams(dmabuf_fd, &parm);

    if (ret != 0)
    {
        cout << "GetParams failed \n";
        return -1;
    }

    void *psrc_data;

    ret = NvBufferMemMap(dmabuf_fd, plane, NvBufferMem_Read_Write, &psrc_data);
    if (ret == 0)
    {
        unsigned int i = 0;
        NvBufferMemSyncForCpu(dmabuf_fd, plane, &psrc_data);
        for (i = 0; i < parm.height[plane]; ++i)
        {
            if(IS_SEMIPLANAR_FMT(parm.pixel_format) && plane == 1)
            {
                stream->write((char *)psrc_data + i * parm.pitch[plane],
                                parm.width[plane] * 2);
                if (!stream->good())
                    return -1;
            }
            else
            {
                stream->write((char *)psrc_data + i * parm.pitch[plane],
                                parm.width[plane]);
                if (!stream->good())
                    return -1;
            }
        }
        NvBufferMemUnMap(dmabuf_fd, plane, &psrc_data);
    }
    else
    {
        cout << "NvBufferMap failed \n";
        return -1;
    }

    return 0;
}

static int
read_decoder_input_nalu(ifstream * stream, NvBuffer * buffer,
        char *parse_buffer, streamsize parse_buffer_size, context_t * ctx)
{
    // Length is the size of the buffer in bytes
    char *buffer_ptr = (char *) buffer->planes[0].data;
    int h265_nal_unit_type;
    char *stream_ptr;
    bool nalu_found = false;

    streamsize bytes_read;
    streamsize stream_initial_pos = stream->tellg();

    stream->read(parse_buffer, parse_buffer_size);
    bytes_read = stream->gcount();

    if (bytes_read == 0)
    {
        return buffer->planes[0].bytesused = 0;
    }

    // Find the first NAL unit in the buffer
    stream_ptr = parse_buffer;
    while ((stream_ptr - parse_buffer) < (bytes_read - 3))
    {
        nalu_found = IS_NAL_UNIT_START(stream_ptr) ||
                    IS_NAL_UNIT_START1(stream_ptr);
        if (nalu_found)
        {
            break;
        }
        stream_ptr++;
    }

    // Reached end of buffer but could not find NAL unit
    if (!nalu_found)
    {
        cerr << "Could not read nal unit from file. EOF or file corrupted"
            << endl;
        return -1;
    }

    memcpy(buffer_ptr, stream_ptr, 4);
    buffer_ptr += 4;
    buffer->planes[0].bytesused = 4;
    stream_ptr += 4;

    if (ctx->copy_timestamp)
    {
      if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H264) {
        if ((IS_H264_NAL_CODED_SLICE(stream_ptr)) ||
            (IS_H264_NAL_CODED_SLICE_IDR(stream_ptr)))
          ctx->flag_copyts = true;
        else
          ctx->flag_copyts = false;
      } else if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H265) {
        h265_nal_unit_type = GET_H265_NAL_UNIT_TYPE(stream_ptr);
        if ((h265_nal_unit_type >= HEVC_NUT_TRAIL_N && h265_nal_unit_type <= HEVC_NUT_RASL_R) ||
            (h265_nal_unit_type >= HEVC_NUT_BLA_W_LP && h265_nal_unit_type <= HEVC_NUT_CRA_NUT))
          ctx->flag_copyts = true;
        else
          ctx->flag_copyts = false;
      }
    }

    // Copy bytes till the next NAL unit is found
    while ((stream_ptr - parse_buffer) < (bytes_read - 3))
    {
        if (IS_NAL_UNIT_START(stream_ptr) || IS_NAL_UNIT_START1(stream_ptr))
        {
            streamsize seekto = stream_initial_pos +
                    (stream_ptr - parse_buffer);
            if(stream->eof())
            {
                stream->clear();
            }
            stream->seekg(seekto, stream->beg);
            return 0;
        }
        *buffer_ptr = *stream_ptr;
        buffer_ptr++;
        stream_ptr++;
        buffer->planes[0].bytesused++;
    }

    // Reached end of buffer but could not find NAL unit
    cerr << "Could not read nal unit from file. EOF or file corrupted"
            << endl;
    return -1;
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
    if(buffer->planes[0].bytesused == 0)
    {
        stream->clear();
        stream->seekg(0,stream->beg);
    }
    return 0;
}

static int
read_vp9_decoder_input_chunk(context_t *ctx, NvBuffer * buffer)
{
    ifstream *stream = ctx->in_file[0];
    int Framesize;
    unsigned char *bitstreambuffer = (unsigned char *)buffer->planes[0].data;
    if (ctx->vp9_file_header_flag == 0)
    {
        stream->read((char *) buffer->planes[0].data, IVF_FILE_HDR_SIZE);
        if (stream->gcount() !=  IVF_FILE_HDR_SIZE)
        {
            cerr << "Couldn't read IVF FILE HEADER" << endl;
            return -1;
        }
        if (!((bitstreambuffer[0] == 'D') && (bitstreambuffer[1] == 'K') &&
                    (bitstreambuffer[2] == 'I') && (bitstreambuffer[3] == 'F')))
        {
            cerr << "It's not a valid IVF file \n" << endl;
            return -1;
        }
        cout << "It's a valid IVF file" << endl;
        ctx->vp9_file_header_flag = 1;
    }
    stream->read((char *) buffer->planes[0].data, IVF_FRAME_HDR_SIZE);
    if (stream->gcount() != IVF_FRAME_HDR_SIZE)
    {
        cerr << "Couldn't read IVF FRAME HEADER" << endl;
        return -1;
    }
    Framesize = (bitstreambuffer[3]<<24) + (bitstreambuffer[2]<<16) +
        (bitstreambuffer[1]<<8) + bitstreambuffer[0];
    buffer->planes[0].bytesused = Framesize;
    stream->read((char *) buffer->planes[0].data, Framesize);
    if (stream->gcount() != Framesize)
    {
        cerr << "Couldn't read Framesize" << endl;
        return -1;
    }
    return 0;
}

static int
read_vp8_decoder_input_chunk(context_t *ctx, NvBuffer * buffer)
{
    ifstream *stream = ctx->in_file[0];
    int Framesize;
    unsigned char *bitstreambuffer = (unsigned char *)buffer->planes[0].data;
    if (ctx->vp8_file_header_flag == 0)
    {
        stream->read((char *) buffer->planes[0].data, IVF_FILE_HDR_SIZE);
        if (stream->gcount() !=  IVF_FILE_HDR_SIZE)
        {
            cerr << "Couldn't read IVF FILE HEADER" << endl;
            return -1;
        }
        if (!((bitstreambuffer[0] == 'D') && (bitstreambuffer[1] == 'K') &&
                    (bitstreambuffer[2] == 'I') && (bitstreambuffer[3] == 'F')))
        {
            cerr << "It's not a valid IVF file \n" << endl;
            return -1;
        }
        cout << "It's a valid IVF file" << endl;
        ctx->vp8_file_header_flag = 1;
    }
    stream->read((char *) buffer->planes[0].data, IVF_FRAME_HDR_SIZE);
    if (stream->gcount() != IVF_FRAME_HDR_SIZE)
    {
        cerr << "Couldn't read IVF FRAME HEADER" << endl;
        return -1;
    }
    Framesize = (bitstreambuffer[3]<<24) + (bitstreambuffer[2]<<16) +
        (bitstreambuffer[1]<<8) + bitstreambuffer[0];
    buffer->planes[0].bytesused = Framesize;
    stream->read((char *) buffer->planes[0].data, Framesize);
    if (stream->gcount() != Framesize)
    {
        cerr << "Couldn't read Framesize" << endl;
        return -1;
    }
    return 0;
}

static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    ctx->dec->abort();
#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx->conv)
    {
        ctx->conv->abort();
        pthread_cond_broadcast(&ctx->queue_cond);
    }
#endif
}

#ifndef USE_NVBUF_TRANSFORM_API
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
    if (ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
        dec_capture_ret_buffer.m.planes[0].m.fd =
            ctx->dmabuff_fd[shared_buffer->index];

    pthread_mutex_lock(&ctx->queue_lock);
    ctx->conv_output_plane_buf_queue->push(buffer);

    // Return the buffer dequeued from converter output plane
    // back to decoder capture plane
    if (ctx->dec->capture_plane.qBuffer(dec_capture_ret_buffer, NULL) < 0)
    {
        abort(ctx);
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
        return false;
    }

    // Write raw video frame to file and return the buffer to converter
    // capture plane
    if (!ctx->stats && ctx->out_file)
    {
        write_video_frame(ctx->out_file, *buffer);
    }

    if (!ctx->stats && !ctx->disable_rendering)
    {
        ctx->renderer->render(buffer->planes[0].fd);
    }

    if (ctx->conv->capture_plane.qBuffer(*v4l2_buf, NULL) < 0)
    {
        return false;
    }
    return true;
}
#endif

static int
report_input_metadata(context_t *ctx, v4l2_ctrl_videodec_inputbuf_metadata *input_metadata)
{
    int ret = -1;
    uint32_t frame_num = ctx->dec->output_plane.getTotalDequeuedBuffers() - 1;

    if (input_metadata->nBitStreamError & V4L2_DEC_ERROR_SPS) {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_SPS " << endl;
    } else if (input_metadata->nBitStreamError & V4L2_DEC_ERROR_PPS) {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_PPS " << endl;
    } else if (input_metadata->nBitStreamError & V4L2_DEC_ERROR_SLICE_HDR) {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_SLICE_HDR " << endl;
    } else if (input_metadata->nBitStreamError & V4L2_DEC_ERROR_MISSING_REF_FRAME) {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_MISSING_REF_FRAME " << endl;
    } else if (input_metadata->nBitStreamError & V4L2_DEC_ERROR_VPS) {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_VPS " << endl;
    } else {
      cout << "Frame " << frame_num << " BitStreamError : ERROR_None " << endl;
      ret = 0;
    }
    return ret;
}

static void
report_metadata(context_t *ctx, v4l2_ctrl_videodec_outputbuf_metadata *metadata)
{
    uint32_t frame_num = ctx->dec->capture_plane.getTotalDequeuedBuffers() - 1;

    cout << "Frame " << frame_num << endl;

    if (metadata->bValidFrameStatus)
    {
        if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H264)
        {
            switch(metadata->CodecParams.H264DecParams.FrameType)
            {
                case 0:
                    cout << "FrameType = B" << endl;
                    break;
                case 1:
                    cout << "FrameType = P" << endl;
                    break;
                case 2:
                    cout << "FrameType = I";
                    if (metadata->CodecParams.H264DecParams.dpbInfo.currentFrame.bIdrFrame)
                    {
                        cout << " (IDR)";
                    }
                    cout << endl;
                    break;
            }
            cout << "nActiveRefFrames = " << metadata->CodecParams.H264DecParams.dpbInfo.nActiveRefFrames << endl;
        }

        if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H265)
        {
            switch(metadata->CodecParams.HEVCDecParams.FrameType)
            {
                case 0:
                    cout << "FrameType = B" << endl;
                    break;
                case 1:
                    cout << "FrameType = P" << endl;
                    break;
                case 2:
                    cout << "FrameType = I";
                    if (metadata->CodecParams.HEVCDecParams.dpbInfo.currentFrame.bIdrFrame)
                    {
                        cout << " (IDR)";
                    }
                    cout << endl;
                    break;
            }
            cout << "nActiveRefFrames = " << metadata->CodecParams.HEVCDecParams.dpbInfo.nActiveRefFrames << endl;
        }

        if (metadata->FrameDecStats.DecodeError)
        {
            v4l2_ctrl_videodec_statusmetadata *dec_stats =
                &metadata->FrameDecStats;
            cout << "ErrorType="  << dec_stats->DecodeError << " Decoded MBs=" <<
                dec_stats->DecodedMBs << " Concealed MBs=" <<
                dec_stats->ConcealedMBs << endl;
        }
    }
    else
    {
        cout << "No valid metadata for frame" << endl;
    }
}

#ifndef USE_NVBUF_TRANSFORM_API
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
#endif

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
    NvBufferCreateParams input_params = {0};
    NvBufferCreateParams cParams = {0};

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
    ctx->display_height = crop.c.height;
    ctx->display_width = crop.c.width;
#ifdef USE_NVBUF_TRANSFORM_API
    if(ctx->dst_dma_fd != -1)
    {
        NvBufferDestroy(ctx->dst_dma_fd);
        ctx->dst_dma_fd = -1;
    }

    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = crop.c.width;
    input_params.height = crop.c.height;
    input_params.layout = NvBufferLayout_Pitch;
    switch(ctx->conv_out_colorspace)
    {
        case 0:
            cout << "Converter output colorspace ITU-R BT.601" << endl;
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12 : NvBufferColorFormat_YUV420;
            else
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12_ER : NvBufferColorFormat_YUV420_ER;
            break;
        case 1:
            cout << "Converter output colorspace ITU-R BT.709" << endl;
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12_709 : NvBufferColorFormat_YUV420_709;
            else
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12_709_ER : NvBufferColorFormat_YUV420_709_ER;
            break;
        case 2:
            cout << "Converter output colorspace ITU-R BT.2020" << endl;
            input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12_2020 : NvBufferColorFormat_YUV420_2020;
            break;
        default:
            cout << "Converter output colorspace ITU-R BT.601" << endl;
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12 : NvBufferColorFormat_YUV420;
            else
                input_params.colorFormat = ctx->out_pixfmt == 1 ? NvBufferColorFormat_NV12_ER : NvBufferColorFormat_YUV420_ER;
            break;
    }

    input_params.nvbuf_tag = NvBufferTag_VIDEO_DEC;

    ret = NvBufferCreateEx (&ctx->dst_dma_fd, &input_params);
    TEST_ERROR(ret == -1, "create dmabuf failed", error);
#else
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
#endif

    if (!ctx->disable_rendering)
    {
        // Destroy the old instance of renderer as resolution might have changed
        delete ctx->renderer;

        if (ctx->fullscreen)
        {
            // Required for fullscreen
            window_width = window_height = 0;
        }
        else if (ctx->window_width && ctx->window_height)
        {
            // As specified by user on commandline
            window_width = ctx->window_width;
            window_height = ctx->window_height;
        }
        else
        {
            // Resolution got from the decoder
            window_width = crop.c.width;
            window_height = crop.c.height;
        }

        // If height or width are set to zero, EglRenderer creates a fullscreen
        // window
        ctx->renderer =
                NvEglRenderer::createEglRenderer("renderer0", window_width,
                                           window_height, ctx->window_x,
                                           ctx->window_y);
        TEST_ERROR(!ctx->renderer,
                   "Error in setting up renderer. "
                   "Check if X is running or run with --disable-rendering",
                   error);
        if (ctx->stats)
        {
            ctx->renderer->enableProfiling();
        }

        ctx->renderer->setFPS(ctx->fps);
    }

    // deinitPlane unmaps the buffers and calls REQBUFS with count 0
    dec->capture_plane.deinitPlane();
    if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
    {
        for(int index = 0 ; index < ctx->numCapBuffers ; index++)
        {
            if(ctx->dmabuff_fd[index] != 0)
            {
                ret = NvBufferDestroy (ctx->dmabuff_fd[index]);
                TEST_ERROR(ret < 0, "Failed to Destroy NvBuffer", error);
            }
        }
    }

    // Not necessary to call VIDIOC_S_FMT on decoder capture plane.
    // But decoder setCapturePlaneFormat function updates the class variables
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

    ctx->video_height = format.fmt.pix_mp.height;
    ctx->video_width = format.fmt.pix_mp.width;
    // Get the minimum buffers which have to be requested on the capture plane
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    TEST_ERROR(ret < 0,
               "Error while getting value of minimum capture plane buffers",
               error);

    // Request (min + 5) buffers, export and map buffers
    if(ctx->capture_plane_mem_type == V4L2_MEMORY_MMAP)
    {
        ret =
            dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                           min_dec_capture_buffers + 5, false,
                                           false);
        TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);
    }
    else if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
    {
        switch(format.fmt.pix_mp.colorspace)
        {
            case V4L2_COLORSPACE_SMPTE170M:
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                {
                    cout << "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12;
                }
                else
                {
                    cout << "Decoder colorspace ITU-R BT.601 with extended range luma (0-255)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12_ER;
                }
                break;
            case V4L2_COLORSPACE_REC709:
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                {
                    cout << "Decoder colorspace ITU-R BT.709 with standard range luma (16-235)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12_709;
                }
                else
                {
                    cout << "Decoder colorspace ITU-R BT.709 with extended range luma (0-255)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12_709_ER;
                }
                break;
            case V4L2_COLORSPACE_BT2020:
                {
                    cout << "Decoder colorspace ITU-R BT.2020" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12_2020;
                }
                break;
            default:
                cout << "supported colorspace details not available, use default" << endl;
                if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                {
                    cout << "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12;
                }
                else
                {
                    cout << "Decoder colorspace ITU-R BT.601 with extended range luma (0-255)" << endl;
                    cParams.colorFormat = NvBufferColorFormat_NV12_ER;
                }
                break;
        }
        ctx->numCapBuffers = min_dec_capture_buffers + 5;
        for (int index = 0; index < ctx->numCapBuffers; index++)
        {
            cParams.width = crop.c.width;
            cParams.height = crop.c.height;
            cParams.layout = NvBufferLayout_BlockLinear;
            cParams.payloadType = NvBufferPayload_SurfArray;
            cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;
            ret = NvBufferCreateEx(&ctx->dmabuff_fd[index], &cParams);
            TEST_ERROR(ret < 0, "Failed to create buffers", error);
        }
        ret = dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,ctx->numCapBuffers);
            TEST_ERROR(ret, "Error in request buffers on capture plane", error);
    }

#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx->conv)
    {
        ret = ctx->conv->setOutputPlaneFormat(format.fmt.pix_mp.pixelformat,
                                              format.fmt.pix_mp.width,
                                              format.fmt.pix_mp.height,
                                              V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR);
        TEST_ERROR(ret < 0, "Error in converter output plane set format",
                   error);

        ret = ctx->conv->setCapturePlaneFormat((ctx->out_pixfmt == 1 ?
                                                    V4L2_PIX_FMT_NV12M :
                                                    V4L2_PIX_FMT_YUV420M),
                                                crop.c.width,
                                                crop.c.height,
                                                V4L2_NV_BUFFER_LAYOUT_PITCH);
        TEST_ERROR(ret < 0, "Error in converter capture plane set format",
                   error);

        ret = ctx->conv->setCropRect(0, 0, crop.c.width, crop.c.height);
        TEST_ERROR(ret < 0, "Error while setting crop rect", error);

        if (ctx->rescale_method) {
            // rescale full range [0-255] to limited range [16-235]
            ret = ctx->conv->setYUVRescale(ctx->rescale_method);
            TEST_ERROR(ret < 0, "Error while setting YUV rescale", error);
        }

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
#endif

    // Capture plane STREAMON
    ret = dec->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

    // Enqueue all the empty capture plane buffers
    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = ctx->capture_plane_mem_type;
        if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
            v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[i];
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

static void *decoder_pollthread_fcn(void *arg)
{

    context_t *ctx = (context_t *) arg;
    v4l2_ctrl_video_device_poll devicepoll;

    cout << "Starting Device Poll Thread " << endl;

    memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

    // wait here until you are signalled to issue the Poll call.
    // Check if the abort status is set , if so exit
    // Else issue the Poll on the decoder and block.
    // When the Poll returns, signal the decoder thread to continue.

    while (!ctx->got_error && !ctx->dec->isInError())
    {
        sem_wait(&ctx->pollthread_sema);

        if (ctx->got_eos)
        {
            cout << "Decoder got eos, exiting poll thread \n";
            return NULL;
        }

        devicepoll.req_events = POLLIN | POLLOUT | POLLERR | POLLPRI;

        // This call shall wait in the v4l2 decoder library
        ctx->dec->DevicePoll(&devicepoll);

        // We can check the devicepoll.resp_events bitmask to see which events are set.
        sem_post(&ctx->decoderthread_sema);
    }
    return NULL;
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
    while ((ev.type != V4L2_EVENT_RESOLUTION_CHANGE) && !ctx->got_error);

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

            if (ctx->enable_metadata)
            {
                v4l2_ctrl_videodec_outputbuf_metadata dec_metadata;

                ret = dec->getMetadata(v4l2_buf.index, dec_metadata);
                if (ret == 0)
                {
                    report_metadata(ctx, &dec_metadata);
                }
            }

            if (ctx->copy_timestamp && ctx->input_nalu && ctx->stats)
            {
              cout << "[" << v4l2_buf.index << "]" "dec capture plane dqB timestamp [" <<
                  v4l2_buf.timestamp.tv_sec << "s" << v4l2_buf.timestamp.tv_usec << "us]" << endl;
            }

            if (!ctx->disable_rendering && ctx->stats)
            {
                // EglRenderer requires the fd of the 0th plane to render the buffer
                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    dec_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];
                ctx->renderer->render(dec_buffer->planes[0].fd);
            }

            // If we need to write to file or display the buffer,
            // give the buffer to video converter output plane
            // instead of returning the buffer back to decoder capture plane
            if (ctx->out_file || (!ctx->disable_rendering && !ctx->stats))
            {
#ifndef USE_NVBUF_TRANSFORM_API
                NvBuffer *conv_buffer;
                struct v4l2_buffer conv_output_buffer;
                struct v4l2_plane conv_planes[MAX_PLANES];

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
                if (ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    dec_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];

                if (ctx->conv->output_plane.
                    qBuffer(conv_output_buffer, dec_buffer) < 0)
                {
                    abort(ctx);
                    cerr <<
                        "Error while queueing buffer at converter output plane"
                        << endl;
                    break;
                }
#else
                /* Clip & Stitch can be done by adjusting rectangle */
                NvBufferRect src_rect, dest_rect;
                src_rect.top = 0;
                src_rect.left = 0;
                src_rect.width = ctx->display_width;
                src_rect.height = ctx->display_height;
                dest_rect.top = 0;
                dest_rect.left = 0;
                dest_rect.width = ctx->display_width;
                dest_rect.height = ctx->display_height;

                NvBufferTransformParams transform_params;
                memset(&transform_params,0,sizeof(transform_params));
                /* Indicates which of the transform parameters are valid */
                transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
                transform_params.transform_flip = NvBufferTransform_None;
                transform_params.transform_filter = NvBufferTransform_Filter_Smart;
                transform_params.src_rect = src_rect;
                transform_params.dst_rect = dest_rect;

                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    dec_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];
                // Convert Blocklinear to PitchLinear
                ret = NvBufferTransform(dec_buffer->planes[0].fd, ctx->dst_dma_fd, &transform_params);
                if (ret == -1)
                {
                    cerr << "Transform failed" << endl;
                    break;
                }

                // Write raw video frame to file
                if (!ctx->stats && ctx->out_file)
                {
                    // Dumping two planes of NV12 and three for I420
                    dump_dmabuf(ctx->dst_dma_fd, 0, ctx->out_file);
                    dump_dmabuf(ctx->dst_dma_fd, 1, ctx->out_file);
                    if (ctx->out_pixfmt != 1)
                    {
                        dump_dmabuf(ctx->dst_dma_fd, 2, ctx->out_file);
                    }
                }

                if (!ctx->stats && !ctx->disable_rendering)
                {
                    ctx->renderer->render(ctx->dst_dma_fd);
                }

                // Not writing to file
                // Queue the buffer back once it has been used.
                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
                {
                    abort(ctx);
                    cerr <<
                        "Error while queueing buffer at decoder capture plane"
                        << endl;
                    break;
                }
#endif
            }
            else
            {
                // Not writing to file
                // Queue the buffer back once it has been used.
                if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
                {
                    abort(ctx);
                    cerr <<
                        "Error while queueing buffer at decoder capture plane"
                        << endl;
                    break;
                }
            }
        }
    }
#ifndef USE_NVBUF_TRANSFORM_API
    // Send EOS to converter
    if (ctx->conv)
    {
        if (sendEOStoConverter(ctx) < 0)
        {
            cerr << "Error while queueing EOS buffer on converter output"
                 << endl;
        }
    }
#endif
    cout << "Exiting decoder capture loop thread" << endl;
    return NULL;
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));
    ctx->fullscreen = false;
    ctx->window_height = 0;
    ctx->window_width = 0;
    ctx->window_x = 0;
    ctx->window_y = 0;
    ctx->out_pixfmt = 1;
    ctx->fps = 30;
    ctx->output_plane_mem_type = V4L2_MEMORY_MMAP;
    ctx->capture_plane_mem_type = V4L2_MEMORY_DMABUF;
    ctx->vp9_file_header_flag = 0;
    ctx->vp8_file_header_flag = 0;
    ctx->stress_test = 1;
    ctx->copy_timestamp = false;
    ctx->flag_copyts = false;
    ctx->start_ts = 0;
    ctx->file_count = 1;
    ctx->dec_fps = 30;
    ctx->dst_dma_fd = -1;
    ctx->bLoop = false;
    ctx->bQueue = false;
    ctx->loop_count = 0;
    ctx->dec_instance_id = V4L2_DEC_INSTANCE_0;
    ctx->blocking_mode = 1;
    ctx->conv_out_colorspace = 0;
#ifndef USE_NVBUF_TRANSFORM_API
    ctx->conv_output_plane_buf_queue = new queue < NvBuffer * >;
    ctx->rescale_method = V4L2_YUV_RESCALE_NONE;
#endif
    pthread_mutex_init(&ctx->queue_lock, NULL);
    pthread_cond_init(&ctx->queue_cond, NULL);
}

static bool decoder_proc_nonblocking(context_t &ctx, bool eos, uint32_t current_file,
                    int current_loop, char *nalu_parse_buffer)
{
    // In non-blocking mode, we will have this function do below things:
    // Issue signal to PollThread so it starts Poll and wait until we are signalled.
    // After we are signalled, it means there is something to dequeue, either output plane
    // or capture plane or there's an event.
    // Try dequeuing from all three and then act appropriately.
    // After enqueuing go back to the same loop.

    // Since all the output plane buffers have been queued, we first need to
    // dequeue a buffer from output plane before we can read new data into it
    // and queue it again.
    int allow_DQ = true;
    int ret = 0;
    struct v4l2_buffer temp_buf;
    struct v4l2_event ev;

    while (!ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_output_buf;
        struct v4l2_plane output_planes[MAX_PLANES];

        struct v4l2_buffer v4l2_capture_buf;
        struct v4l2_plane capture_planes[MAX_PLANES];

        NvBuffer *output_buffer = NULL;
        NvBuffer *capture_buffer = NULL;

        memset(&v4l2_output_buf, 0, sizeof(v4l2_output_buf));
        memset(output_planes, 0, sizeof(output_planes));
        v4l2_output_buf.m.planes = output_planes;

        memset(&v4l2_capture_buf, 0, sizeof(v4l2_capture_buf));
        memset(capture_planes, 0, sizeof(capture_planes));
        v4l2_capture_buf.m.planes = capture_planes;

        // Call SetPollInterrupt
        ctx.dec->SetPollInterrupt();

        // Since buffers have been queued, issue a post to start polling and then wait here
        sem_post(&ctx.pollthread_sema);
        sem_wait(&ctx.decoderthread_sema);

        ret = ctx.dec->dqEvent(ev, 0);
        if (ret == 0)
        {
            if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE)
            {
                cout << "Got V4L2_EVENT_RESOLUTION_CHANGE EVENT \n";
                query_and_set_capture(&ctx);
            }
        }

        while (1)
        {
            // Now dequeue from the output plane and enqueue back the buffers after reading
            if ( (eos) && (ctx.dec->output_plane.getNumQueuedBuffers() == 0) )
            {
                cout << "Done processing all the buffers returning \n";
                return true;
            }

            if (allow_DQ)
            {
                ret = ctx.dec->output_plane.dqBuffer(v4l2_output_buf, &output_buffer, NULL, 0);
                if (ret < 0)
                {
                    if (errno == EAGAIN)
                        goto check_capture_buffers;
                    else
                    {
                        cerr << "Error DQing buffer at output plane" << endl;
                        abort(&ctx);
                        break;
                    }
                }
            }
            else
            {
                allow_DQ = true;
                memcpy(&v4l2_output_buf,&temp_buf,sizeof(v4l2_buffer));
                output_buffer = ctx.dec->output_plane.getNthBuffer(v4l2_output_buf.index);
            }

            if ((v4l2_output_buf.flags & V4L2_BUF_FLAG_ERROR) && ctx.enable_input_metadata)
            {
                v4l2_ctrl_videodec_inputbuf_metadata dec_input_metadata;

                ret = ctx.dec->getInputMetadata(v4l2_output_buf.index, dec_input_metadata);
                if (ret == 0)
                {
                    ret = report_input_metadata(&ctx, &dec_input_metadata);
                    if (ret == -1)
                    {
                        cerr << "Error with input stream header parsing" << endl;
                    }
                }
            }

            if (eos)
            {
                //cout << "Got EOS , no more queueing of buffers on OUTPUT plane \n";
                goto check_capture_buffers;
            }

            if ((ctx.decoder_pixfmt == V4L2_PIX_FMT_H264) ||
                    (ctx.decoder_pixfmt == V4L2_PIX_FMT_H265) ||
                    (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG2) ||
                    (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG4))
            {
                if (ctx.input_nalu)
                {
                    read_decoder_input_nalu(ctx.in_file[current_file], output_buffer, nalu_parse_buffer,
                            CHUNK_SIZE, &ctx);
                }
                else
                {
                    read_decoder_input_chunk(ctx.in_file[current_file], output_buffer);
                }
            }
            if (ctx.decoder_pixfmt == V4L2_PIX_FMT_VP9)
            {
                ret = read_vp9_decoder_input_chunk(&ctx, output_buffer);
                if (ret != 0)
                    cerr << "Couldn't read VP9 chunk" << endl;
            }
            v4l2_output_buf.m.planes[0].bytesused = output_buffer->planes[0].bytesused;

            if (ctx.input_nalu && ctx.copy_timestamp && ctx.flag_copyts)
            {
                v4l2_output_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
                ctx.timestamp += ctx.timestampincr;
                v4l2_output_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
                v4l2_output_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
            }

            if (v4l2_output_buf.m.planes[0].bytesused == 0)
            {
                if (ctx.bQueue)
                {
                    current_file++;
                    if(current_file != ctx.file_count)
                    {
                        allow_DQ = false;
                        memcpy(&temp_buf, &v4l2_output_buf, sizeof(v4l2_buffer));
                        continue;
                    }
                }
                if(ctx.bLoop)
                {
                    current_file = current_file % ctx.file_count;
                    allow_DQ = false;
                    memcpy(&temp_buf, &v4l2_output_buf,sizeof(v4l2_buffer));
                    if (ctx.loop_count == 0 || current_loop < ctx.loop_count )
                    {
                        current_loop++;
                        continue;
                    }
                }
            }
            ret = ctx.dec->output_plane.qBuffer(v4l2_output_buf, NULL);
            if (ret < 0)
            {
                cerr << "Error Qing buffer at output plane" << endl;
                abort(&ctx);
                break;
            }
            if (v4l2_output_buf.m.planes[0].bytesused == 0)
            {
                eos = true;
                cout << "Input file read complete" << endl;
                goto check_capture_buffers;
            }
        }

        // Dequeue from the capture plane and write them to file and enqueue back
check_capture_buffers:
        while (1)
        {
            if (!ctx.dec->capture_plane.getStreamStatus())
            {
                cout << "Capture plane not ON, skipping capture plane \n";
                break;
            }
            // Dequeue a filled buffer
            ret = ctx.dec->capture_plane.dqBuffer(v4l2_capture_buf, &capture_buffer, NULL, 0);
            if (ret < 0)
            {
                if (errno == EAGAIN)
                    break;
                else
                {
                    abort(&ctx);
                    cerr << "Error while calling dequeue at capture plane" <<
                        endl;
                }
                break;
            }
            if (capture_buffer == NULL)
            {
                cout << "Got CAPTURE BUFFER NULL \n";
                break;
            }

            if (ctx.enable_metadata)
            {
                v4l2_ctrl_videodec_outputbuf_metadata dec_metadata;

                ret = ctx.dec->getMetadata(v4l2_capture_buf.index, dec_metadata);
                if (ret == 0)
                {
                    report_metadata(&ctx, &dec_metadata);
                }
            }

            if (ctx.copy_timestamp && ctx.input_nalu && ctx.stats)
            {
              cout << "[" << v4l2_capture_buf.index << "]" "dec capture plane dqB timestamp [" <<
                  v4l2_capture_buf.timestamp.tv_sec << "s" << v4l2_capture_buf.timestamp.tv_usec << "us]" << endl;
            }

            if (!ctx.disable_rendering && ctx.stats)
            {
                // Rendering the buffer here
                // EglRenderer requires the fd of the 0th plane to render the buffer
                if(ctx.capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    capture_buffer->planes[0].fd = ctx.dmabuff_fd[v4l2_capture_buf.index];
                //cout << "Enqueue the buffer to renderer " << capture_buffer->planes[0].fd << endl;
                if (ctx.renderer->render(capture_buffer->planes[0].fd) == -1)
                {
                    abort(&ctx);
                    cerr << "Error while queueing buffer for rendering "
                            << endl;
                    break;
                }
            }

            if (ctx.out_file || (!ctx.disable_rendering && !ctx.stats))
            {
                NvBufferRect src_rect, dest_rect;
                src_rect.top = 0;
                src_rect.left = 0;
                src_rect.width = ctx.display_width;
                src_rect.height = ctx.display_height;
                dest_rect.top = 0;
                dest_rect.left = 0;
                dest_rect.width = ctx.display_width;
                dest_rect.height = ctx.display_height;

                NvBufferTransformParams transform_params;
                /* Indicates which of the transform parameters are valid */
                memset(&transform_params, 0, sizeof(transform_params));
                transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
                transform_params.transform_flip = NvBufferTransform_None;
                transform_params.transform_filter = NvBufferTransform_Filter_Smart;
                transform_params.src_rect = src_rect;
                transform_params.dst_rect = dest_rect;

                if(ctx.capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    capture_buffer->planes[0].fd = ctx.dmabuff_fd[v4l2_capture_buf.index];
                // Convert Blocklinear to PitchLinear
                ret = NvBufferTransform(capture_buffer->planes[0].fd, ctx.dst_dma_fd, &transform_params);
                if (ret == -1)
                {
                    cerr << "Transform failed" << endl;
                    break;
                }
                // Write raw video frame to file
                if (!ctx.stats && ctx.out_file)
                {
                    // Dumping two planes of NV12 and three for I420
                    cout << "Writing to file \n";
                    dump_dmabuf(ctx.dst_dma_fd, 0, ctx.out_file);
                    dump_dmabuf(ctx.dst_dma_fd, 1, ctx.out_file);
                    if (ctx.out_pixfmt != 1)
                    {
                        dump_dmabuf(ctx.dst_dma_fd, 2, ctx.out_file);
                    }
                }
                if (!ctx.stats && !ctx.disable_rendering)
                {
                    ctx.renderer->render(ctx.dst_dma_fd);
                }
                // Queue the buffer back once it has been used.
                // If we are not rendering, queue the buffer back here immediately.
                if(ctx.capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    v4l2_capture_buf.m.planes[0].m.fd = ctx.dmabuff_fd[v4l2_capture_buf.index];
                if (ctx.dec->capture_plane.qBuffer(v4l2_capture_buf, NULL) < 0)
                {
                    abort(&ctx);
                    cerr << "Error while queueing buffer at decoder capture plane"
                            << endl;
                    break;
                }
            }
        }
    }
    return eos;
}

static bool decoder_proc_blocking(context_t &ctx, bool eos, uint32_t current_file,
                                int current_loop, char *nalu_parse_buffer)
{
    // Since all the output plane buffers have been queued, we first need to
    // dequeue a buffer from output plane before we can read new data into it
    // and queue it again.
    int allow_DQ = true;
    int ret = 0;
    struct v4l2_buffer temp_buf;

    while (!eos && !ctx.got_error && !ctx.dec->isInError())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        if(allow_DQ)
        {
            ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
            if (ret < 0)
            {
                cerr << "Error DQing buffer at output plane" << endl;
                abort(&ctx);
                break;
            }
        }
        else
        {
            allow_DQ = true;
            memcpy(&v4l2_buf,&temp_buf,sizeof(v4l2_buffer));
            buffer = ctx.dec->output_plane.getNthBuffer(v4l2_buf.index);
        }

        if ((v4l2_buf.flags & V4L2_BUF_FLAG_ERROR) && ctx.enable_input_metadata)
        {
            v4l2_ctrl_videodec_inputbuf_metadata dec_input_metadata;

            ret = ctx.dec->getInputMetadata(v4l2_buf.index, dec_input_metadata);
            if (ret == 0)
            {
                ret = report_input_metadata(&ctx, &dec_input_metadata);
                if (ret == -1)
                {
                  cerr << "Error with input stream header parsing" << endl;
                }
            }
        }

        if ((ctx.decoder_pixfmt == V4L2_PIX_FMT_H264) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_H265) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG2) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG4))
        {
            if (ctx.input_nalu)
            {
                read_decoder_input_nalu(ctx.in_file[current_file], buffer, nalu_parse_buffer,
                        CHUNK_SIZE, &ctx);
            }
            else
            {
                read_decoder_input_chunk(ctx.in_file[current_file], buffer);
            }
        }
        if (ctx.decoder_pixfmt == V4L2_PIX_FMT_VP9)
        {
            ret = read_vp9_decoder_input_chunk(&ctx, buffer);
            if (ret != 0)
                cerr << "Couldn't read VP9 chunk" << endl;
        }
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        if (ctx.input_nalu && ctx.copy_timestamp && ctx.flag_copyts)
        {
          v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
          ctx.timestamp += ctx.timestampincr;
          v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
          v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            if (ctx.bQueue)
            {
                current_file++;
                if(current_file != ctx.file_count)
                {
                    allow_DQ = false;
                    memcpy(&temp_buf,&v4l2_buf,sizeof(v4l2_buffer));
                    continue;
                }
            }
            if(ctx.bLoop)
            {
                current_file = current_file % ctx.file_count;
                allow_DQ = false;
                memcpy(&temp_buf,&v4l2_buf,sizeof(v4l2_buffer));
            }
        }
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
    return eos;
}

static int
decode_proc(context_t& ctx, int argc, char *argv[])
{
    int ret = 0;
    int error = 0;
    uint32_t current_file = 0;
    uint32_t i;
    bool eos = false;
    int current_loop = 0;
    char *nalu_parse_buffer = NULL;
    NvApplicationProfiler &profiler = NvApplicationProfiler::getProfilerInstance();

    set_defaults(&ctx);

    if (parse_csv_args(&ctx, argc, argv))
    {
        fprintf(stderr, "Error parsing commandline arguments\n");
        return -1;
    }
    if (ctx.blocking_mode)
    {
        cout << "Creating decoder in blocking mode \n";
        ctx.dec = NvVideoDecoder::createVideoDecoder("dec0");
    }
    else
    {
        cout << "Creating decoder in non-blocking mode \n";
        ctx.dec = NvVideoDecoder::createVideoDecoder("dec0", O_NONBLOCK);
    }
    TEST_ERROR(!ctx.dec, "Could not create decoder", cleanup);

    if (ctx.stats)
    {
        profiler.start(NvApplicationProfiler::DefaultSamplingInterval);
        ctx.dec->enableProfiling();
    }

    // Subscribe to Resolution change event
    ret = ctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE",
               cleanup);

    // Set format on the output plane
    ret = ctx.dec->setOutputPlaneFormat(ctx.decoder_pixfmt, CHUNK_SIZE);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    if (ctx.input_nalu)
    {
        nalu_parse_buffer = new char[CHUNK_SIZE];
        printf("Setting frame input mode to 0 \n");
        ret = ctx.dec->setFrameInputMode(0);
        TEST_ERROR(ret < 0,
                "Error in decoder setFrameInputMode", cleanup);
    }
    else
    {
        // Set V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control to false
        // so that application can send chunks of encoded data instead of forming
        // complete frames.
        printf("Setting frame input mode to 1 \n");
        ret = ctx.dec->setFrameInputMode(1);
        TEST_ERROR(ret < 0,
                "Error in decoder setFrameInputMode", cleanup);
    }

    // V4L2_CID_MPEG_VIDEO_DISABLE_DPB should be set after output plane
    // set format
    if (ctx.disable_dpb)
    {
        ret = ctx.dec->disableDPB();
        TEST_ERROR(ret < 0, "Error in decoder disableDPB", cleanup);
    }

    if (ctx.enable_metadata || ctx.enable_input_metadata)
    {
        ret = ctx.dec->enableMetadataReporting();
        TEST_ERROR(ret < 0, "Error while enabling metadata reporting", cleanup);
    }

    if (ctx.dec_instance_id)
    {
        ret = ctx.dec->setDecInstance(ctx.dec_instance_id);
        TEST_ERROR(ret < 0, "Error while setting decoder instance id", cleanup);
    }

    if (ctx.skip_frames)
    {
        ret = ctx.dec->setSkipFrames(ctx.skip_frames);
        TEST_ERROR(ret < 0, "Error while setting skip frames param", cleanup);
    }

    // Query, Export and Map the output plane buffers so that we can read
    // encoded data into the buffers
    if (ctx.output_plane_mem_type == V4L2_MEMORY_MMAP)
        ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    else if (ctx.output_plane_mem_type == V4L2_MEMORY_USERPTR)
        ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);

    TEST_ERROR(ret < 0, "Error while setting up output plane", cleanup);

    ctx.in_file = (std::ifstream **)malloc(sizeof(std::ifstream *)*ctx.file_count);
    for (uint32_t i = 0 ; i < ctx.file_count ; i++)
    {
        ctx.in_file[i] = new ifstream(ctx.in_file_path[i]);
        TEST_ERROR(!ctx.in_file[i]->is_open(), "Error opening input file", cleanup);
    }

    if (ctx.out_file_path)
    {
        ctx.out_file = new ofstream(ctx.out_file_path);
        TEST_ERROR(!ctx.out_file->is_open(), "Error opening output file",
                   cleanup);
    }

#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx.out_file || (!ctx.disable_rendering && !ctx.stats))
    {
        // Create converter to convert from BL to PL for writing raw video
        // to file
        ctx.conv = NvVideoConverter::createVideoConverter("conv0");
        TEST_ERROR(!ctx.conv, "Could not create video converter", cleanup);
        ctx.conv->output_plane.
            setDQThreadCallback(conv0_output_dqbuf_thread_callback);
        ctx.conv->capture_plane.
            setDQThreadCallback(conv0_capture_dqbuf_thread_callback);

        if (ctx.stats)
        {
            ctx.conv->enableProfiling();
        }
    }
#endif

    ret = ctx.dec->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane stream on", cleanup);

    if (ctx.blocking_mode)
    {
        pthread_create(&ctx.dec_capture_loop, NULL, dec_capture_loop_fcn, &ctx);
    }
    else
    {
        sem_init(&ctx.pollthread_sema, 0, 0);
        sem_init(&ctx.decoderthread_sema, 0, 0);
        pthread_create(&ctx.dec_pollthread, NULL, decoder_pollthread_fcn, &ctx);
        cout << "Created the PollThread and Decoder Thread \n";
    }

    if (ctx.copy_timestamp && ctx.input_nalu) {
      ctx.timestamp = (ctx.start_ts * MICROSECOND_UNIT);
      ctx.timestampincr = (MICROSECOND_UNIT * 16) / ((uint32_t) (ctx.dec_fps * 16));
    }

    // Read encoded data and enqueue all the output plane buffers.
    // Exit loop in case file read is complete.
    i = 0;
    current_loop = 1;
    while (!eos && !ctx.got_error && !ctx.dec->isInError() &&
           i < ctx.dec->output_plane.getNumBuffers())
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        buffer = ctx.dec->output_plane.getNthBuffer(i);
        if ((ctx.decoder_pixfmt == V4L2_PIX_FMT_H264) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_H265) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG2) ||
                (ctx.decoder_pixfmt == V4L2_PIX_FMT_MPEG4))
        {
            if (ctx.input_nalu)
            {
                read_decoder_input_nalu(ctx.in_file[current_file], buffer, nalu_parse_buffer,
                        CHUNK_SIZE, &ctx);
            }
            else
            {
                read_decoder_input_chunk(ctx.in_file[current_file], buffer);
            }
        }
        if (ctx.decoder_pixfmt == V4L2_PIX_FMT_VP9)
        {
            ret = read_vp9_decoder_input_chunk(&ctx, buffer);
            if (ret != 0)
                cerr << "Couldn't read VP9 chunk" << endl;
        }
        if (ctx.decoder_pixfmt == V4L2_PIX_FMT_VP8)
        {
            ret = read_vp8_decoder_input_chunk(&ctx, buffer);
            if (ret != 0)
                cerr << "Couldn't read VP8 chunk" << endl;
        }
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        if (ctx.input_nalu && ctx.copy_timestamp && ctx.flag_copyts)
        {
          v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
          ctx.timestamp += ctx.timestampincr;
          v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
          v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            if (ctx.bQueue)
            {
                current_file++;
                if(current_file != ctx.file_count)
                {
                    continue;
                }
            }
            if(ctx.bLoop)
            {
                current_file = current_file % ctx.file_count;
                if(ctx.loop_count == 0 || current_loop < ctx.loop_count )
                {
                    current_loop++;
                    continue;
                }
            }
        }
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
    if (ctx.blocking_mode)
        eos = decoder_proc_blocking(ctx, eos, current_file, current_loop, nalu_parse_buffer);
    else
        eos = decoder_proc_nonblocking(ctx, eos, current_file, current_loop, nalu_parse_buffer);
    // After sending EOS, all the buffers from output plane should be dequeued.
    // and after that capture plane loop should be signalled to stop.
    if (ctx.blocking_mode)
    {
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

            if ((v4l2_buf.flags & V4L2_BUF_FLAG_ERROR) && ctx.enable_input_metadata)
            {
                v4l2_ctrl_videodec_inputbuf_metadata dec_input_metadata;

                ret = ctx.dec->getInputMetadata(v4l2_buf.index, dec_input_metadata);
                if (ret == 0)
                {
                    ret = report_input_metadata(&ctx, &dec_input_metadata);
                    if (ret == -1)
                    {
                      cerr << "Error with input stream header parsing" << endl;
                      abort(&ctx);
                      break;
                    }
                }
            }
        }
    }

    // Signal EOS to the decoder capture loop
    ctx.got_eos = true;
#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx.conv)
    {
        ctx.conv->capture_plane.waitForDQThread(-1);
    }
#endif

    if (ctx.stats)
    {
        profiler.stop();
        ctx.dec->printProfilingStats(cout);
#ifndef USE_NVBUF_TRANSFORM_API
        if (ctx.conv)
        {
            ctx.conv->printProfilingStats(cout);
        }
#endif
        if (ctx.renderer)
        {
            ctx.renderer->printProfilingStats(cout);
        }
        profiler.printProfilerData(cout);
    }

cleanup:
    if (ctx.blocking_mode && ctx.dec_capture_loop)
    {
        pthread_join(ctx.dec_capture_loop, NULL);
    }
    else if (!ctx.blocking_mode)
    {
        // Clear the poll interrupt to get the decoder's poll thread out.
        ctx.dec->ClearPollInterrupt();
        // If Pollthread is waiting on, signal it to exit the thread.
        sem_post(&ctx.pollthread_sema);
        pthread_join(ctx.dec_pollthread, NULL);
    }
    if(ctx.capture_plane_mem_type == V4L2_MEMORY_DMABUF)
    {
        for(int index = 0 ; index < ctx.numCapBuffers ; index++)
        {
            if(ctx.dmabuff_fd[index] != 0)
            {
                ret = NvBufferDestroy (ctx.dmabuff_fd[index]);
                if(ret < 0)
                {
                    cerr << "Failed to Destroy NvBuffer" << endl;
                }
            }
        }
    }
#ifndef USE_NVBUF_TRANSFORM_API
    if (ctx.conv && ctx.conv->isInError())
    {
        cerr << "Converter is in error" << endl;
        error = 1;
    }
#endif
    if (ctx.dec && ctx.dec->isInError())
    {
        cerr << "Decoder is in error" << endl;
        error = 1;
    }

    if (ctx.got_error)
    {
        error = 1;
    }

    // The decoder destructor does all the cleanup i.e set streamoff on output and capture planes,
    // unmap buffers, tell decoder to deallocate buffer (reqbufs ioctl with counnt = 0),
    // and finally call v4l2_close on the fd.
    delete ctx.dec;
#ifndef USE_NVBUF_TRANSFORM_API
    delete ctx.conv;
#endif
    // Similarly, EglRenderer destructor does all the cleanup
    delete ctx.renderer;
    for (uint32_t i = 0 ; i < ctx.file_count ; i++)
      delete ctx.in_file[i];
    delete ctx.out_file;
#ifndef USE_NVBUF_TRANSFORM_API
    delete ctx.conv_output_plane_buf_queue;
#else
    if(ctx.dst_dma_fd != -1)
    {
        NvBufferDestroy(ctx.dst_dma_fd);
        ctx.dst_dma_fd = -1;
    }
#endif
    delete[] nalu_parse_buffer;

    free (ctx.in_file);
    for (uint32_t i = 0 ; i < ctx.file_count ; i++)
      free (ctx.in_file_path[i]);
    free (ctx.in_file_path);
    free(ctx.out_file_path);
    if (!ctx.blocking_mode)
    {
        sem_destroy(&ctx.pollthread_sema);
        sem_destroy(&ctx.decoderthread_sema);
    }

    return -error;
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int ret = 0;
    int iterator_num = 0; //save decode iterator number

    do
    {
        ret = decode_proc(ctx, argc, argv);
        iterator_num++;
    } while((ctx.stress_test != iterator_num) && ret == 0);

    if (ret)
    {
        cout << "App run failed" << endl;
    }
    else
    {
        cout << "App run was successful" << endl;
    }

    return ret;
}
