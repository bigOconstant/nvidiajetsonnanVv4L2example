/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include "NvUtils.h"
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <sstream>
#include <string.h>
#include <fcntl.h>
#include <poll.h>
#include "nvbuf_utils.h"

#include "video_encode.h"

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define TEST_PARSE_ERROR(cond, label) if(cond) { \
    cerr << "Error parsing runtime parameter changes string" << endl; \
    goto label; }

#define IS_DIGIT(c) (c >= '0' && c <= '9')
#define MICROSECOND_UNIT 1000000

using namespace std;

static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    ctx->enc->abort();
}

//Initialise CRC Rec and creates CRC Table based on the polynomial.
static
Crc* InitCrc(unsigned int CrcPolynomial)
{
    unsigned short int i;
    unsigned short int j;
    unsigned int tempcrc;
    Crc *phCrc;
    phCrc = (Crc*) malloc (sizeof(Crc));
    if (phCrc == NULL)
    {
        cerr << "Mem allocation failed for Init CRC" <<endl;
        return NULL;
    }

    memset (phCrc, 0, sizeof(Crc));

    for (i = 0; i <= 255; i++)
    {
        tempcrc = i;
        for (j = 8; j > 0; j--)
        {
            if (tempcrc & 1)
            {
                tempcrc = (tempcrc >> 1) ^ CrcPolynomial;
            }
            else
            {
                tempcrc >>= 1;
            }
        }
        phCrc->CRCTable[i] = tempcrc;
    }

    phCrc->CrcValue = 0;
    return phCrc;
}

//Calculates CRC of data provided in by buffer.
static
void CalculateCrc(Crc *phCrc, unsigned char *buffer, uint32_t count)
{
    unsigned char *p;
    unsigned int temp1;
    unsigned int temp2;
    unsigned int crc = phCrc->CrcValue;
    unsigned int *CRCTable = phCrc->CRCTable;

    if(!count)
        return;

    p = (unsigned char *) buffer;
    while (count-- != 0)
    {
        temp1 = (crc >> 8) & 0x00FFFFFFL;
        temp2 = CRCTable[((unsigned int) crc ^ *p++) & 0xFF];
        crc = temp1 ^ temp2;
    }

    phCrc->CrcValue = crc;
}

//Closes CRC related handles.
static
void CloseCrc(Crc **phCrc)
{
    if (*phCrc)
        free (*phCrc);
}

static int
write_encoder_output_frame(ofstream * stream, NvBuffer * buffer)
{
    stream->write((char *) buffer->planes[0].data, buffer->planes[0].bytesused);
    return 0;
}

static bool
encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
                                  NvBuffer * shared_buffer, void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvVideoEncoder *enc = ctx->enc;
    uint32_t frame_num = ctx->enc->capture_plane.getTotalDequeuedBuffers() - 1;
    uint32_t ReconRef_Y_CRC = 0;
    uint32_t ReconRef_U_CRC = 0;
    uint32_t ReconRef_V_CRC = 0;
    static uint32_t num_encoded_frames = 1;
    struct v4l2_event ev;
    int ret = 0;

    if (v4l2_buf == NULL)
    {
        cout << "Error while dequeing buffer from output plane" << endl;
        abort(ctx);
        return false;
    }

    if (ctx->b_use_enc_cmd)
    {
        if(v4l2_buf->flags & V4L2_BUF_FLAG_LAST)
        {
            memset(&ev,0,sizeof(struct v4l2_event));
            ret = ctx->enc->dqEvent(ev,1000);
            if (ret < 0)
                cout << "Error in dqEvent" << endl;
            if(ev.type == V4L2_EVENT_EOS)
                return false;
        }
    }

    // GOT EOS from encoder. Stop dqthread.
    if (buffer->planes[0].bytesused == 0)
    {
        cout << "Got 0 size buffer in capture \n";
        return false;
    }

    // Computing CRC with each frame
    if(ctx->pBitStreamCrc)
        CalculateCrc (ctx->pBitStreamCrc, buffer->planes[0].data, buffer->planes[0].bytesused);

    write_encoder_output_frame(ctx->out_file, buffer);
    num_encoded_frames++;

    // Accounting for the first frame as it is only sps+pps
    if (ctx->gdr_out_frame_number != 0xFFFFFFFF)
        if ( (ctx->enableGDR) && (ctx->GDR_out_file_path) && (num_encoded_frames >= ctx->gdr_out_frame_number+1))
            write_encoder_output_frame(ctx->gdr_out_file, buffer);

    if (ctx->report_metadata)
    {
        v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
        if (ctx->enc->getMetadata(v4l2_buf->index, enc_metadata) == 0)
        {
            if (ctx->bReconCrc && enc_metadata.bValidReconCRC) {
                // CRC for Recon frame
                cout << "Frame: " << frame_num << endl;
                cout << "ReconFrame_Y_CRC " << enc_metadata.ReconFrame_Y_CRC <<
                    " ReconFrame_U_CRC " << enc_metadata.ReconFrame_U_CRC <<
                    " ReconFrame_V_CRC " << enc_metadata.ReconFrame_V_CRC <<
                    endl;

                if (!ctx->recon_Ref_file->eof()) {
                    *ctx->recon_Ref_file >> ReconRef_Y_CRC;
                    *ctx->recon_Ref_file >> ReconRef_U_CRC;
                    *ctx->recon_Ref_file >> ReconRef_V_CRC;
                }

                if ((ReconRef_Y_CRC != enc_metadata.ReconFrame_Y_CRC) ||
                    (ReconRef_U_CRC != enc_metadata.ReconFrame_U_CRC) ||
                    (ReconRef_V_CRC != enc_metadata.ReconFrame_V_CRC))
                {
                    cout << "Recon CRC FAIL" << endl;
                    cout << "ReconRef_Y_CRC " << ReconRef_Y_CRC <<
                        " ReconRef_U_CRC " << ReconRef_U_CRC <<
                        " ReconRef_V_CRC " << ReconRef_V_CRC <<
                        endl;
                    abort(ctx);
                    return false;
                }
                cout << "Recon CRC PASS for frame : " << frame_num << endl;
            } else if (ctx->externalRPS && enc_metadata.bRPSFeedback_status) {
                // RPS Feedback
                cout << "Frame: " << frame_num << endl;
                cout << "nCurrentRefFrameId " << enc_metadata.nCurrentRefFrameId <<
                     " nActiveRefFrames " << enc_metadata.nActiveRefFrames << endl;

                for (uint32_t i = 0; i < enc_metadata.nActiveRefFrames; i++)
                {
                    cout << "FrameId " << enc_metadata.RPSList[i].nFrameId <<
                     " IdrFrame " <<  (int) enc_metadata.RPSList[i].bIdrFrame <<
                     " LTRefFrame " <<  (int) enc_metadata.RPSList[i].bLTRefFrame <<
                     " PictureOrderCnt " << enc_metadata.RPSList[i].nPictureOrderCnt <<
                     " FrameNum " << enc_metadata.RPSList[i].nFrameNum <<
                     " LTFrameIdx " <<  enc_metadata.RPSList[i].nLTRFrameIdx << endl;
                }
            } else if (ctx->externalRCHints) {
                // Rate Control Feedback
                cout << "Frame: " << frame_num << endl;
                cout << "EncodedBits " << enc_metadata.EncodedFrameBits <<
                    " MinQP " << enc_metadata.FrameMinQP <<
                    " MaxQP " << enc_metadata.FrameMaxQP <<
                    endl;
            } else {
                cout << "Frame " << frame_num <<
                    ": isKeyFrame=" << (int) enc_metadata.KeyFrame <<
                    " AvgQP=" << enc_metadata.AvgQP <<
                    " MinQP=" << enc_metadata.FrameMinQP <<
                    " MaxQP=" << enc_metadata.FrameMaxQP <<
                    " EncodedBits=" << enc_metadata.EncodedFrameBits <<
                    endl;
            }
        }
    }
    if (ctx->dump_mv)
    {
        v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata;
        if (ctx->enc->getMotionVectors(v4l2_buf->index, enc_mv_metadata) == 0)
        {
            uint32_t numMVs = enc_mv_metadata.bufSize / sizeof(MVInfo);
            MVInfo *pInfo = enc_mv_metadata.pMVInfo;

            cout << "Frame " << frame_num << ": Num MVs=" << numMVs << endl;

            for (uint32_t i = 0; i < numMVs; i++, pInfo++)
            {
                cout << i << ": mv_x=" << pInfo->mv_x <<
                    " mv_y=" << pInfo->mv_y <<
                    " weight=" << pInfo->weight <<
                    endl;
            }
        }
    }

    if (enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0)
    {
        cerr << "Error while Qing buffer at capture plane" << endl;
        abort(ctx);
        return false;
    }

    return true;
}

static int
get_next_parsed_pair(context_t *ctx, char *id, uint32_t *value)
{
    char charval;

    *ctx->runtime_params_str >> *id;
    if (ctx->runtime_params_str->eof())
    {
        return -1;
    }

    charval = ctx->runtime_params_str->peek();
    if (!IS_DIGIT(charval))
    {
        return -1;
    }

    *ctx->runtime_params_str >> *value;

    *ctx->runtime_params_str >> charval;
    if (ctx->runtime_params_str->eof())
    {
        return 0;
    }

    return charval;
}

static int
set_runtime_params(context_t *ctx)
{
    char charval;
    uint32_t intval;
    int ret, next;

    cout << "Frame " << ctx->next_param_change_frame <<
        ": Changing parameters" << endl;
    while (!ctx->runtime_params_str->eof())
    {
        next = get_next_parsed_pair(ctx, &charval, &intval);
        TEST_PARSE_ERROR(next < 0, err);
        switch (charval)
        {
            case 'b':
                if (ctx->ratecontrol == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR &&
                    ctx->peak_bitrate < intval) {
                    uint32_t peak_bitrate = 1.2f * intval;
                    cout << "Peak bitrate = " << peak_bitrate << endl;
                    ret = ctx->enc->setPeakBitrate(peak_bitrate);
                    if (ret < 0)
                    {
                        cerr << "Could not set encoder peakbitrate" << endl;
                        goto err;
                    }
                }
                cout << "Bitrate = " << intval << endl;
                ret = ctx->enc->setBitrate(intval);
                if (ret < 0)
                {
                    cerr << "Could not set encoder bitrate" << endl;
                    goto err;
                }
                break;
            case 'p':
                cout << "Peak bitrate = " << intval << endl;
                ret = ctx->enc->setPeakBitrate(intval);
                if (ret < 0)
                {
                    cerr << "Could not set encoder peakbitrate" << endl;
                    goto err;
                }
                break;
            case 'r':
            {
                int fps_num = intval;
                TEST_PARSE_ERROR(next != '/', err);

                ctx->runtime_params_str->seekg(-1, ios::cur);
                next = get_next_parsed_pair(ctx, &charval, &intval);
                TEST_PARSE_ERROR(next < 0, err);

                cout << "Framerate = " << fps_num << "/"  << intval << endl;

                ret = ctx->enc->setFrameRate(fps_num, intval);
                if (ret < 0)
                {
                    cerr << "Could not set framerate" << endl;
                    goto err;
                }
                break;
            }
            case 'i':
                if (intval > 0)
                {
                    ctx->enc->forceIDR();
                    cout << "Forcing IDR" << endl;
                }
                break;
            default:
                TEST_PARSE_ERROR(true, err);
        }
        switch (next)
        {
            case 0:
                delete ctx->runtime_params_str;
                ctx->runtime_params_str = NULL;
                return 0;
            case '#':
                return 0;
            case ',':
                break;
            default:
                break;
        }
    }
    return 0;
err:
    cerr << "Skipping further runtime parameter changes" <<endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

static int
get_next_runtime_param_change_frame(context_t *ctx)
{
    char charval;
    int ret;

    ret = get_next_parsed_pair(ctx, &charval, &ctx->next_param_change_frame);
    if(ret == 0)
    {
        return 0;
    }

    TEST_PARSE_ERROR((ret != ';' && ret != ',') || charval != 'f', err);

    return 0;

err:
    cerr << "Skipping further runtime parameter changes" <<endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->bitrate = 4 * 1024 * 1024;
    ctx->peak_bitrate = 0;
    ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
    ctx->ratecontrol = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
    ctx->iframe_interval = 30;
    ctx->externalRPS = false;
    ctx->enableGDR = false;
    ctx->enableROI = false;
    ctx->bnoIframe = false;
    ctx->bGapsInFrameNumAllowed = false;
    ctx->bReconCrc = false;
    ctx->enableLossless = false;
    ctx->nH264FrameNumBits = 0;
    ctx->nH265PocLsbBits = 0;
    ctx->idr_interval = 256;
    ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
    ctx->fps_n = 30;
    ctx->fps_d = 1;
    ctx->gdr_start_frame_number = 0xffffffff;
    ctx->gdr_num_frames = 0xffffffff;
    ctx->gdr_out_frame_number = 0xffffffff;
    ctx->num_b_frames = (uint32_t) -1;
    ctx->nMinQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->use_gold_crc = false;
    ctx->pBitStreamCrc = NULL;
    ctx->externalRCHints = false;
    ctx->input_metadata = false;
    ctx->sMaxQp = 51;
    ctx->stress_test = 1;
    ctx->output_memory_type = V4L2_MEMORY_DMABUF;
    ctx->copy_timestamp = false;
    ctx->start_ts = 0;
    ctx->blocking_mode = 1;
}

static void
populate_roi_Param(std::ifstream * stream, v4l2_enc_frame_ROI_params *VEnc_ROI_params)
{
    unsigned int ROIIndex = 0;

    if (!stream->eof()) {
        *stream >> VEnc_ROI_params->num_ROI_regions;
        while (ROIIndex < VEnc_ROI_params->num_ROI_regions)
        {
            if (ROIIndex == V4L2_MAX_ROI_REGIONS) {
                string skip_str;
                getline(*stream, skip_str);

                VEnc_ROI_params->num_ROI_regions = V4L2_MAX_ROI_REGIONS;

                cout << "Maximum of " << V4L2_MAX_ROI_REGIONS <<
                        "regions can be applied for a frame" << endl;
                break;
            }

            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].QPdelta;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.left;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.top;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.width;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.height;
            ROIIndex++;
        }
    } else {
        cout << "EOF of ROI_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
}

static void
populate_ext_rps_ctrl_Param (std::ifstream * stream, v4l2_enc_frame_ext_rps_ctrl_params *VEnc_ext_rps_ctrl_params)
{
    unsigned int RPSIndex = 0;
    unsigned int temp = 0;

    stream->peek();
restart :
    if (stream->eof()) {
        cout << "EOF of rps_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        *stream >> VEnc_ext_rps_ctrl_params->nFrameId;
        if (stream->eof())
            goto restart;
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bRefFrame = ((temp)?true:false);
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bLTRefFrame = ((temp)?true:false);
        *stream >> VEnc_ext_rps_ctrl_params->nMaxRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nActiveRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nCurrentRefFrameId;
        while (RPSIndex < VEnc_ext_rps_ctrl_params->nActiveRefFrames)
        {
            if (RPSIndex == V4L2_MAX_REF_FRAMES) {
                string skip_str;
                getline(*stream, skip_str);

                VEnc_ext_rps_ctrl_params->nActiveRefFrames = V4L2_MAX_REF_FRAMES;

                cout << "Maximum of " << V4L2_MAX_REF_FRAMES <<
                        "reference frames are valid" << endl;
                break;
            }

            *stream >> VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].nFrameId;
            *stream >> temp;
            VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].bLTRefFrame = ((temp)?true:false);
            RPSIndex++;
        }
    }
}

static int
setup_output_dmabuf(context_t *ctx, uint32_t num_buffers )
{
    int ret=0;
    NvBufferCreateParams cParams;
    int fd;
    ret = ctx->enc->output_plane.reqbufs(V4L2_MEMORY_DMABUF,num_buffers);
    if(ret)
    {
        cerr << "reqbufs failed for output plane V4L2_MEMORY_DMABUF" << endl;
        return ret;
    }
    for (uint32_t i = 0; i < ctx->enc->output_plane.getNumBuffers(); i++)
    {
        cParams.width = ctx->width;
        cParams.height = ctx->height;
        cParams.layout = NvBufferLayout_Pitch;
        cParams.colorFormat = ctx->enable_extended_colorformat ?
             NvBufferColorFormat_YUV420_ER : NvBufferColorFormat_YUV420;
        cParams.nvbuf_tag = NvBufferTag_VIDEO_ENC;
        cParams.payloadType = NvBufferPayload_SurfArray;
        ret = NvBufferCreateEx(&fd, &cParams);
        if(ret < 0)
        {
            cerr << "Failed to create NvBuffer" << endl;
            return ret;
        }
        ctx->output_plane_fd[i]=fd;
    }
    return ret;
}

static void
populate_ext_rate_ctrl_Param(std::ifstream * stream, v4l2_enc_frame_ext_rate_ctrl_params *VEnc_ext_rate_ctrl_params)
{
    stream->peek();
restart:
    if (stream->eof()) {
        cout << "EOF of hints_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        *stream >> VEnc_ext_rate_ctrl_params->nTargetFrameBits;
        if (stream->eof())
            goto restart;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameQP;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMinQp;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMaxQp;
        *stream >> VEnc_ext_rate_ctrl_params->nMaxQPDeviation;
    }
}

static void
populate_gdr_Param(std::ifstream * stream, uint32_t *start_frame_num, uint32_t *gdr_num_frames)
{
    if (stream->eof()) {
        *start_frame_num = 0xFFFFFFFF;
        cout << "GDR param EoF reached \n";
    }
    if (!stream->eof()) {
        *stream >> *start_frame_num;
        *stream >> *gdr_num_frames;
    }
}


static void *encoder_pollthread_fcn(void *arg)
{

    context_t *ctx = (context_t *) arg;
    v4l2_ctrl_video_device_poll devicepoll;

    cout << "Starting Device Poll Thread " << endl;

    memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

    // wait here until you are signalled to issue the Poll call.
    // Check if the abort status is set , if so exit
    // Else issue the Poll on the decoder and block.
    // When the Poll returns, signal the encoder thread to continue.

    while (!ctx->got_error && !ctx->enc->isInError())
    {
        sem_wait(&ctx->pollthread_sema);

        if (ctx->got_eos)
        {
            cout << "Got eos, exiting poll thread \n";
            return NULL;
        }

        devicepoll.req_events = POLLIN | POLLOUT | POLLERR | POLLPRI;

        // This call shall wait in the v4l2 encoder library
        ctx->enc->DevicePoll(&devicepoll);

        // We can check the devicepoll.resp_events bitmask to see which events are set.
        sem_post(&ctx->encoderthread_sema);
    }
    return NULL;
}

static int encoder_proc_nonblocking(context_t &ctx, bool eos)
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
    int ret = 0;

    while (!ctx.got_error && !ctx.enc->isInError())
    {
        // Call SetPollInterrupt
        ctx.enc->SetPollInterrupt();

        // Since buffers have been queued, issue a post to start polling and then wait here
        sem_post(&ctx.pollthread_sema);
        sem_wait(&ctx.encoderthread_sema);

        // Already end of file, no more queue-dequeu for output plane
        if (eos)
            goto check_capture_buffers;

        //Lets first check if we can dequeue from output plane
        while (1)
        {

            struct v4l2_buffer v4l2_output_buf;
            struct v4l2_plane output_planes[MAX_PLANES];
            NvBuffer *outplane_buffer = NULL;

            memset(&v4l2_output_buf, 0, sizeof(v4l2_output_buf));
            memset(output_planes, 0, sizeof(output_planes));
            v4l2_output_buf.m.planes = output_planes;

            // Dequeue from output plane, fill the frame and enqueue it back again.
            // Could be moved out to a different thread as an optimization.
            ret = ctx.enc->output_plane.dqBuffer(v4l2_output_buf, &outplane_buffer, NULL, 10);
            if (ret < 0)
            {
                if (errno == EAGAIN)
                {
                    goto check_capture_buffers;
                }
                cerr << "ERROR while DQing buffer at output plane" << endl;
                abort(&ctx);
                return -1;
            }

            if (ctx.runtime_params_str &&
                    (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                     ctx.next_param_change_frame))
            {
                set_runtime_params(&ctx);
                if (ctx.runtime_params_str)
                    get_next_runtime_param_change_frame(&ctx);
            }
            if (read_video_frame(ctx.in_file, *outplane_buffer) < 0)
            {
                cerr << "Could not read complete frame from input file" << endl;
                v4l2_output_buf.m.planes[0].bytesused = 0;
                if(ctx.b_use_enc_cmd)
                {
                    ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                    eos = true;
                    break;
                }
                else
                {
                    eos = true;
                    v4l2_output_buf.m.planes[0].m.userptr = 0;
                    v4l2_output_buf.m.planes[0].bytesused = 0;
                    v4l2_output_buf.m.planes[1].bytesused = 0;
                    v4l2_output_buf.m.planes[2].bytesused = 0;
                }
            }

            if (ctx.input_metadata)
            {
                v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
                v4l2_enc_frame_ROI_params VEnc_ROI_params;
                v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
                v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
                v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
                v4l2_enc_gdr_params VEnc_gdr_params;
                VEnc_imeta_param.flag = 0;

                if (ctx.ROI_Param_file_path)
                {
                    if (ctx.enableROI) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;
                        populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                    }
                }

                if (ctx.bReconCrc)
                {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                    VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                    VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                    VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                    VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                    VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
                }

                if (ctx.RPS_Param_file_path)
                {
                    if (ctx.externalRPS) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                        populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                    }
                }

                if (ctx.GDR_Param_file_path)
                {
                    if (ctx.enableGDR)
                    {
                        if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                            populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                    &ctx.gdr_num_frames);
                        if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number)
                        {
                            ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                            VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                            VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                            VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                        }
                    }
                }

                if (ctx.hints_Param_file_path)
                {
                    if (ctx.externalRCHints) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                        populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);

                    }
                }

                if (VEnc_imeta_param.flag)
                {
                    ctx.enc->SetInputMetaParams(v4l2_output_buf.index, VEnc_imeta_param);
                    v4l2_output_buf.reserved2 = v4l2_output_buf.index;
                }
            }

            if (ctx.copy_timestamp)
            {
                v4l2_output_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
                ctx.timestamp += ctx.timestampincr;
                v4l2_output_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
                v4l2_output_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
            }

            if(ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP)
            {
                for (int j = 0 ; j < MAX_PLANES ; j++)
                {
                    ret = NvBufferMemSyncForDevice (outplane_buffer->planes[j].fd, j, (void **)&outplane_buffer->planes[j].data);
                    if (ret < 0)
                    {
                        cerr << "Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                        abort(&ctx);
                        return -1;
                    }
                }
            }

            if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
            {
                for (int j = 0 ; j < MAX_PLANES ; j++)
                {
                    v4l2_output_buf.m.planes[j].bytesused = outplane_buffer->planes[j].bytesused;
                }
            }
            ret = ctx.enc->output_plane.qBuffer(v4l2_output_buf, NULL);
            if (ret < 0)
            {
                cerr << "Error while queueing buffer at output plane" << endl;
                abort(&ctx);
                return -1;
            }

            ctx.input_frames_queued_count++;
            if (v4l2_output_buf.m.planes[0].bytesused == 0)
            {
                cerr << "File read complete." << endl;
                eos = true;
                goto check_capture_buffers;
            }
        }

check_capture_buffers:
        while (1)
        {
            struct v4l2_buffer v4l2_capture_buf;
            struct v4l2_plane capture_planes[MAX_PLANES];
            NvBuffer *capplane_buffer = NULL;
            bool capture_dq_continue = true;

            memset(&v4l2_capture_buf, 0, sizeof(v4l2_capture_buf));
            memset(capture_planes, 0, sizeof(capture_planes));
            v4l2_capture_buf.m.planes = capture_planes;
            v4l2_capture_buf.length = 1;

            // Dequeue from output plane, fill the frame and enqueue it back again.
            // Could be moved out to a different thread as an optimization.
            ret = ctx.enc->capture_plane.dqBuffer(v4l2_capture_buf, &capplane_buffer, NULL, 10);
            if (ret < 0)
            {
                if (errno == EAGAIN)
                    break;
                cerr << "ERROR while DQing buffer at capture plane" << endl;
                abort(&ctx);
                return -1;
            }

            capture_dq_continue = encoder_capture_plane_dq_callback(&v4l2_capture_buf, capplane_buffer, NULL,
                    &ctx);
            if (!capture_dq_continue)
            {
                cout << "Capture plane dequeued 0 size buffer " << endl;
                ctx.got_eos = true;
                return 0;
            }
        }
    }

    return 0;
}

static int encoder_proc_blocking(context_t &ctx, bool eos)
{
    int ret = 0;
    // Keep reading input till EOS is reached
    while (!ctx.got_error && !ctx.enc->isInError() && !eos)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        if (ctx.enc->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10) < 0)
        {
            cerr << "ERROR while DQing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                ctx.next_param_change_frame))
        {
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }
        if (read_video_frame(ctx.in_file, *buffer) < 0)
        {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
            if(ctx.b_use_enc_cmd)
            {
                ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                eos = true;
                ctx.got_eos = true;
                break;
            }
            else
            {
                eos = true;
                ctx.got_eos = true;
                v4l2_buf.m.planes[0].m.userptr = 0;
                v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
            }
        }

        if (ctx.input_metadata)
        {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            v4l2_enc_gdr_params VEnc_gdr_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path)
            {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;
                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc)
            {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.RPS_Param_file_path)
            {
                if (ctx.externalRPS) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.GDR_Param_file_path)
            {
                if (ctx.enableGDR)
                {
                    if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                        populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                    &ctx.gdr_num_frames);
                    if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number)
                    {
                        ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                        VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                    }
                }
            }

            if (ctx.hints_Param_file_path)
            {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);

                }
            }

            if (VEnc_imeta_param.flag)
            {
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        if (ctx.copy_timestamp)
        {
          v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
          ctx.timestamp += ctx.timestampincr;
          v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
          v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if(ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP)
        {
            for (int j = 0 ; j < MAX_PLANES ; j++)
            {
                ret = NvBufferMemSyncForDevice (buffer->planes[j].fd, j, (void **)&buffer->planes[j].data);
                if (ret < 0)
                {
                    cerr << "Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
            }
        }

        if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
        {
            for (int j = 0 ; j < MAX_PLANES ; j++)
            {
                v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
            }
        }
        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        ctx.input_frames_queued_count++;
        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            cerr << "File read complete." << endl;
            eos = true;
            ctx.got_eos = true;
            return 0;
        }
    }
cleanup:
    return -1;
}

static int
encode_proc(context_t& ctx, int argc, char *argv[])
{
    int ret = 0;
    int error = 0;
    bool eos = false;

    set_defaults(&ctx);

    ret = parse_csv_args(&ctx, argc, argv);
    TEST_ERROR(ret < 0, "Error parsing commandline arguments", cleanup);

    if (ctx.runtime_params_str)
    {
        get_next_runtime_param_change_frame(&ctx);
    }

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265)
    {
        TEST_ERROR(ctx.width < 144 || ctx.height < 144, "Height/Width should be"
            " > 144 for H.265", cleanup);
    }

    if (ctx.use_gold_crc)
    {
        ctx.pBitStreamCrc = InitCrc(CRC32_POLYNOMIAL);
        TEST_ERROR(!ctx.pBitStreamCrc, "InitCrc failed", cleanup);
    }

    ctx.in_file = new ifstream(ctx.in_file_path);
    TEST_ERROR(!ctx.in_file->is_open(), "Could not open input file", cleanup);

    ctx.out_file = new ofstream(ctx.out_file_path);
    TEST_ERROR(!ctx.out_file->is_open(), "Could not open output file", cleanup);

    if (ctx.ROI_Param_file_path) {
        ctx.roi_Param_file = new ifstream(ctx.ROI_Param_file_path);
        TEST_ERROR(!ctx.roi_Param_file->is_open(), "Could not open roi param file", cleanup);
    }

    if (ctx.Recon_Ref_file_path) {
        ctx.recon_Ref_file = new ifstream(ctx.Recon_Ref_file_path);
        TEST_ERROR(!ctx.recon_Ref_file->is_open(), "Could not open recon crc reference file", cleanup);
    }

    if (ctx.RPS_Param_file_path) {
        ctx.rps_Param_file = new ifstream(ctx.RPS_Param_file_path);
        TEST_ERROR(!ctx.rps_Param_file->is_open(), "Could not open rps param file", cleanup);
    }

    if (ctx.GDR_Param_file_path) {
        ctx.gdr_Param_file = new ifstream(ctx.GDR_Param_file_path);
        TEST_ERROR(!ctx.gdr_Param_file->is_open(), "Could not open GDR param file", cleanup);
    }

    if (ctx.GDR_out_file_path) {
        ctx.gdr_out_file = new ofstream(ctx.GDR_out_file_path);
        TEST_ERROR(!ctx.gdr_out_file->is_open(), "Could not open GDR Out file", cleanup);
    }

    if (ctx.hints_Param_file_path) {
        ctx.hints_Param_file = new ifstream(ctx.hints_Param_file_path);
        TEST_ERROR(!ctx.hints_Param_file->is_open(), "Could not open hints param file", cleanup);
    }

    if (ctx.blocking_mode)
    {
        cout << "Creating Encoder in blocking mode \n";
        ctx.enc = NvVideoEncoder::createVideoEncoder("enc0");
    }
    else
    {
        cout << "Creating Encoder in non-blocking mode \n";
        ctx.enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);
    }
    TEST_ERROR(!ctx.enc, "Could not create encoder", cleanup);

    // It is necessary that Capture Plane format be set before Output Plane
    // format.
    // Set encoder capture plane format. It is necessary to set width and
    // height on thr capture plane as well
    ret =
        ctx.enc->setCapturePlaneFormat(ctx.encoder_pixfmt, ctx.width,
                                      ctx.height, 2 * 1024 * 1024);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    // Set encoder output plane format
    ret =
        ctx.enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, ctx.width,
                                      ctx.height);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    ret = ctx.enc->setBitrate(ctx.bitrate);
    TEST_ERROR(ret < 0, "Could not set encoder bitrate", cleanup);

    ret = ctx.enc->setProfile(ctx.profile);
    TEST_ERROR(ret < 0, "Could not set encoder profile", cleanup);

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264)
    {
        ret = ctx.enc->setLevel(ctx.level);
        TEST_ERROR(ret < 0, "Could not set encoder level", cleanup);
    }

    if (ctx.enableLossless)
    {
        ret = ctx.enc->setConstantQp(0);
        TEST_ERROR(ret < 0, "Could not set encoder constant qp=0", cleanup);
    }
    else
    {
        ret = ctx.enc->setRateControlMode(ctx.ratecontrol);
        TEST_ERROR(ret < 0, "Could not set encoder rate control mode", cleanup);
        if (ctx.ratecontrol == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR) {
            uint32_t peak_bitrate;
            if (ctx.peak_bitrate < ctx.bitrate)
                peak_bitrate = 1.2f * ctx.bitrate;
            else
                peak_bitrate = ctx.peak_bitrate;
            ret = ctx.enc->setPeakBitrate(peak_bitrate);
            TEST_ERROR(ret < 0, "Could not set encoder peak bitrate", cleanup);
        }
    }

    ret = ctx.enc->setIDRInterval(ctx.idr_interval);
    TEST_ERROR(ret < 0, "Could not set encoder IDR interval", cleanup);

    ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
    TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);

    ret = ctx.enc->setFrameRate(ctx.fps_n, ctx.fps_d);
    TEST_ERROR(ret < 0, "Could not set framerate", cleanup);

    if (ctx.temporal_tradeoff_level)
    {
        ret = ctx.enc->setTemporalTradeoff(ctx.temporal_tradeoff_level);
        TEST_ERROR(ret < 0, "Could not set temporal tradeoff level", cleanup);
    }

    if (ctx.slice_length)
    {
        ret = ctx.enc->setSliceLength(ctx.slice_length_type,
                ctx.slice_length);
        TEST_ERROR(ret < 0, "Could not set slice length params", cleanup);
    }

    if (ctx.hw_preset_type)
    {
        ret = ctx.enc->setHWPresetType(ctx.hw_preset_type);
        TEST_ERROR(ret < 0, "Could not set encoder HW Preset Type", cleanup);
    }

    if (ctx.virtual_buffer_size)
    {
        ret = ctx.enc->setVirtualBufferSize(ctx.virtual_buffer_size);
        TEST_ERROR(ret < 0, "Could not set virtual buffer size", cleanup);
    }

    if (ctx.num_reference_frames)
    {
        ret = ctx.enc->setNumReferenceFrames(ctx.num_reference_frames);
        TEST_ERROR(ret < 0, "Could not set num reference frames", cleanup);
    }

    if (ctx.slice_intrarefresh_interval)
    {
        ret = ctx.enc->setSliceIntrarefresh(ctx.slice_intrarefresh_interval);
        TEST_ERROR(ret < 0, "Could not set slice intrarefresh interval", cleanup);
    }

    if (ctx.insert_sps_pps_at_idr)
    {
        ret = ctx.enc->setInsertSpsPpsAtIdrEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertSPSPPSAtIDR", cleanup);
    }

    if (ctx.insert_vui)
    {
        ret = ctx.enc->setInsertVuiEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertVUI", cleanup);
    }

    if (ctx.enable_extended_colorformat)
    {
        ret = ctx.enc->setExtendedColorFormat(true);
        TEST_ERROR(ret < 0, "Could not set extended color format", cleanup);
    }

    if (ctx.insert_aud)
    {
        ret = ctx.enc->setInsertAudEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertAUD", cleanup);
    }

    if (ctx.num_b_frames != (uint32_t) -1)
    {
        ret = ctx.enc->setNumBFrames(ctx.num_b_frames);
        TEST_ERROR(ret < 0, "Could not set number of B Frames", cleanup);
    }

    if ((ctx.nMinQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpB != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpB != (uint32_t)QP_RETAIN_VAL))
    {
        ret = ctx.enc->setQpRange(ctx.nMinQpI, ctx.nMaxQpI, ctx.nMinQpP,
                ctx.nMaxQpP, ctx.nMinQpB, ctx.nMaxQpB);
        TEST_ERROR(ret < 0, "Could not set quantization parameters", cleanup);
    }

    if (ctx.dump_mv)
    {
        ret = ctx.enc->enableMotionVectorReporting();
        TEST_ERROR(ret < 0, "Could not enable motion vector reporting", cleanup);
    }

    if (ctx.bnoIframe) {
        ctx.iframe_interval = ((1<<31) + 1); /* TODO: how can we do this properly */
        ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
        TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);
    }

    if (ctx.enableROI) {
        v4l2_enc_enable_roi_param VEnc_enable_ext_roi_ctrl;

        VEnc_enable_ext_roi_ctrl.bEnableROI = ctx.enableROI;
        ret = ctx.enc->enableROI(VEnc_enable_ext_roi_ctrl);
        TEST_ERROR(ret < 0, "Could not enable ROI", cleanup);
    }

    if (ctx.bReconCrc) {
        v4l2_enc_enable_reconcrc_param VEnc_enable_recon_crc_ctrl;

        VEnc_enable_recon_crc_ctrl.bEnableReconCRC = ctx.bReconCrc;
        ret = ctx.enc->enableReconCRC(VEnc_enable_recon_crc_ctrl);
        TEST_ERROR(ret < 0, "Could not enable Recon CRC", cleanup);
    }

    if (ctx.externalRPS) {
        v4l2_enc_enable_ext_rps_ctr VEnc_enable_ext_rps_ctrl;

        VEnc_enable_ext_rps_ctrl.bEnableExternalRPS = ctx.externalRPS;
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
            VEnc_enable_ext_rps_ctrl.bGapsInFrameNumAllowed = ctx.bGapsInFrameNumAllowed;
            VEnc_enable_ext_rps_ctrl.nH264FrameNumBits = ctx.nH264FrameNumBits;
        }
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {
            VEnc_enable_ext_rps_ctrl.nH265PocLsbBits = ctx.nH265PocLsbBits;
        }
        ret = ctx.enc->enableExternalRPS(VEnc_enable_ext_rps_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RPS", cleanup);
    }

    if (ctx.externalRCHints) {
        v4l2_enc_enable_ext_rate_ctr VEnc_enable_ext_rate_ctrl;

        VEnc_enable_ext_rate_ctrl.bEnableExternalPictureRC = ctx.externalRCHints;
        VEnc_enable_ext_rate_ctrl.nsessionMaxQP = ctx.sMaxQp;

        ret = ctx.enc->enableExternalRC(VEnc_enable_ext_rate_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RC", cleanup);
    }

    // Query, Export and Map the output plane buffers so that we can read
    // raw data into the buffers
    switch(ctx.output_memory_type)
    {
        case V4L2_MEMORY_MMAP:
            ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
            TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
            break;

        case V4L2_MEMORY_USERPTR:
            ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
            TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
            break;

        case V4L2_MEMORY_DMABUF:
            ret = setup_output_dmabuf(&ctx,10);
            TEST_ERROR(ret < 0, "Could not setup plane", cleanup);
            break;
        default :
            TEST_ERROR(true, "Not a valid plane", cleanup);
    }

    ret = ctx.enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Could not setup capture plane", cleanup);

    ret = ctx.enc->subscribeEvent(V4L2_EVENT_EOS,0,0);
    TEST_ERROR(ret < 0, "Could not subscribe EOS event", cleanup);

    if (ctx.b_use_enc_cmd)
    {
        ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_START, 0);
        TEST_ERROR(ret < 0, "Error in start of encoder commands ", cleanup);
    }
    else
    {
        // output plane STREAMON
        ret = ctx.enc->output_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in output plane streamon", cleanup);

        // capture plane STREAMON
        ret = ctx.enc->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in capture plane streamon", cleanup);
    }


    if (ctx.blocking_mode)
    {
        ctx.enc->capture_plane.
            setDQThreadCallback(encoder_capture_plane_dq_callback);

        // startDQThread starts a thread internally which calls the
        // encoder_capture_plane_dq_callback whenever a buffer is dequeued
        // on the plane
        ctx.enc->capture_plane.startDQThread(&ctx);
    }
    else
    {
        sem_init(&ctx.pollthread_sema, 0, 0);
        sem_init(&ctx.encoderthread_sema, 0, 0);
        pthread_create(&ctx.enc_pollthread, NULL, encoder_pollthread_fcn, &ctx);
        cout << "Created the PollThread and Encoder Thread \n";
    }

    // Enqueue all the empty capture plane buffers
    for (uint32_t i = 0; i < ctx.enc->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        ret = ctx.enc->capture_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at capture plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
    }

    if (ctx.copy_timestamp) {
      ctx.timestamp = (ctx.start_ts * MICROSECOND_UNIT);
      ctx.timestampincr = (MICROSECOND_UNIT * 16) / ((uint32_t) (ctx.fps_n * 16));
    }

    // Read video frame and queue all the output plane buffers
    for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer = ctx.enc->output_plane.getNthBuffer(i);

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
        {
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            ret = ctx.enc->output_plane.mapOutputBuffers(v4l2_buf, ctx.output_plane_fd[i]);

            if (ret < 0)
            {
                cerr << "Error while mapping buffer at output plane" << endl;
                abort(&ctx);
                goto cleanup;
            }
        }

        if (read_video_frame(ctx.in_file, *buffer) < 0)
        {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
            if(ctx.b_use_enc_cmd)
             {
                ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                eos = true;
                break;
             }
             else
             {
                 eos = true;
                 v4l2_buf.m.planes[0].m.userptr = 0;
                 v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
             }
        }

        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                ctx.next_param_change_frame))
        {
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }

        if (ctx.input_metadata)
        {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            v4l2_enc_gdr_params VEnc_gdr_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path)
            {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;

                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc)
            {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.RPS_Param_file_path)
            {
                if (ctx.externalRPS) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;

                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.GDR_Param_file_path)
            {
                if (ctx.enableGDR)
                {
                    if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                        populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                    &ctx.gdr_num_frames);
                    if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number)
                    {
                        ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                        VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                    }
                }
            }

            if (ctx.hints_Param_file_path)
            {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);
                }
            }

            if (VEnc_imeta_param.flag)
            {
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        if (ctx.copy_timestamp)
        {
          v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
          ctx.timestamp += ctx.timestampincr;
          v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
          v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if(ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP)
        {
            for (int j = 0 ; j < MAX_PLANES ; j++)
            {
                ret = NvBufferMemSyncForDevice (buffer->planes[j].fd, j, (void **)&buffer->planes[j].data);
                if (ret < 0)
                {
                    cerr << "Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
            }
        }

        if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
        {
            for (int j = 0 ; j < MAX_PLANES ; j++)
            {
                v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
            }
        }
        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            cerr << "File read complete." << endl;
            eos = true;
            break;
        }
        ctx.input_frames_queued_count++;
    }

    if (ctx.blocking_mode)
    {
        // Wait till capture plane DQ Thread finishes
        // i.e. all the capture plane buffers are dequeued
        if (encoder_proc_blocking(ctx, eos) != 0)
            goto cleanup;
        ctx.enc->capture_plane.waitForDQThread(-1);
    }
    else
    {
        if (encoder_proc_nonblocking(ctx, eos) != 0)
            goto cleanup;
    }

cleanup:
    if (ctx.enc && ctx.enc->isInError())
    {
        cerr << "Encoder is in error" << endl;
        error = 1;
    }
    if (ctx.got_error)
    {
        error = 1;
    }

    if (ctx.pBitStreamCrc)
    {
        char *pgold_crc = ctx.gold_crc;
        Crc *pout_crc= ctx.pBitStreamCrc;
        char StrCrcValue[20];
        snprintf (StrCrcValue, 20, "%u", pout_crc->CrcValue);
        // Remove CRLF from end of CRC, if present
        do {
               unsigned int len = strlen(pgold_crc);
               if (len == 0) break;
               if (pgold_crc[len-1] == '\n')
                   pgold_crc[len-1] = '\0';
               else if (pgold_crc[len-1] == '\r')
                   pgold_crc[len-1] = '\0';
               else
                   break;
        } while(1);

        if (strcmp (StrCrcValue, pgold_crc))
        {
            cout << "======================" << endl;
            cout << "video_encode: CRC FAILED" << endl;
            cout << "======================" << endl;
            cout << "Encoded CRC: " << StrCrcValue << " Gold CRC: " << pgold_crc << endl;
            error = 1;
        }
        else
        {
            cout << "======================" << endl;
            cout << "video_encode: CRC PASSED" << endl;
            cout << "======================" << endl;
        }

        CloseCrc(&ctx.pBitStreamCrc);
    }

    if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
    {
        for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++)
        {
            ret = ctx.enc->output_plane.unmapOutputBuffers(i, ctx.output_plane_fd[i]);
            if (ret < 0)
            {
                cerr << "Error while unmapping buffer at output plane" << endl;
                goto cleanup;
            }

            ret = NvBufferDestroy(ctx.output_plane_fd[i]);
            if(ret < 0)
            {
                cerr << "Failed to Destroy NvBuffer\n" << endl;
                return ret;
            }
        }
    }

    delete ctx.enc;
    delete ctx.in_file;
    delete ctx.out_file;
    delete ctx.roi_Param_file;
    delete ctx.recon_Ref_file;
    delete ctx.rps_Param_file;
    delete ctx.hints_Param_file;
    delete ctx.gdr_Param_file;
    delete ctx.gdr_out_file;

    free(ctx.in_file_path);
    free(ctx.out_file_path);
    free(ctx.ROI_Param_file_path);
    free(ctx.Recon_Ref_file_path);
    free(ctx.RPS_Param_file_path);
    free(ctx.hints_Param_file_path);
    free(ctx.GDR_Param_file_path);
    free(ctx.GDR_out_file_path);
    delete ctx.runtime_params_str;

    if (!ctx.blocking_mode)
    {
        sem_destroy(&ctx.pollthread_sema);
        sem_destroy(&ctx.encoderthread_sema);
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
        ret = encode_proc(ctx, argc, argv);
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
