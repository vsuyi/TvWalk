#include <iostream>
#include <thread>
#include <chrono>
#include <Windows.h>

#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}


const char* filter_descr = "select='gt(scene,0.11)'";

static AVFormatContext* fmt_ctx;
static AVCodecContext* dec_ctx;
AVFilterContext* buffersink_ctx;
AVFilterContext* buffersrc_ctx;
AVFilterGraph* filter_graph;
static int video_stream_index = -1;
static int64_t last_pts = AV_NOPTS_VALUE;

static cv::Mat AVFrame_to_CvMat(const AVFrame* input_avframe)
{
    int image_width = input_avframe->width;
    int image_height = input_avframe->height;

    cv::Mat resMat(image_height, image_width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = resMat.step1();

    SwsContext* avFrameToOpenCVBGRSwsContext = sws_getContext(
        image_width,
        image_height,
        (AVPixelFormat)input_avframe->format,
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );

    sws_scale(avFrameToOpenCVBGRSwsContext,
        input_avframe->data,
        input_avframe->linesize,
        0,
        image_height,
        &resMat.data,
        cvLinesizes);

    if (avFrameToOpenCVBGRSwsContext != nullptr)
    {
        sws_freeContext(avFrameToOpenCVBGRSwsContext);
        avFrameToOpenCVBGRSwsContext = nullptr;
    }

    cv::Mat resizedMat(image_height / 2, image_width / 2, CV_8UC3);
    cv::resize(resMat, resizedMat, cv::Size(image_width / 2, image_height / 2));

    return resizedMat;
}

static int open_input_file(const char* filename)
{
    const AVCodec* dec;
    int ret;

    if ((ret = avformat_open_input(&fmt_ctx, filename, NULL, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open input file\n");
        return ret;
    }

    if ((ret = avformat_find_stream_info(fmt_ctx, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
        return ret;
    }

    /* select the video stream */
    ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find a video stream in the input file\n");
        return ret;
    }
    video_stream_index = ret;

    /* create decoding context */
    dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx)
        return AVERROR(ENOMEM);
    avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_stream_index]->codecpar);

    /* init the video decoder */
    if ((ret = avcodec_open2(dec_ctx, dec, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open video decoder\n");
        return ret;
    }

    return 0;
}

static int init_filters(const char* filters_descr)
{
    char args[512];
    int ret = 0;
    const AVFilter* buffersrc = avfilter_get_by_name("buffer");
    const AVFilter* buffersink = avfilter_get_by_name("buffersink");
    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs = avfilter_inout_alloc();
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE };

    filter_graph = avfilter_graph_alloc();
    if (!outputs || !inputs || !filter_graph) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    /* buffer video source: the decoded frames from the decoder will be inserted here. */
    snprintf(args, sizeof(args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        time_base.num, time_base.den,
        dec_ctx->sample_aspect_ratio.num, dec_ctx->sample_aspect_ratio.den);

    ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in",
        args, NULL, filter_graph);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create buffer source\n");
        goto end;
    }

    /* buffer video sink: to terminate the filter chain. */
    ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
        NULL, NULL, filter_graph);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create buffer sink\n");
        goto end;
    }

    ret = av_opt_set_int_list(buffersink_ctx, "pix_fmts", pix_fmts,
        AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot set output pixel format\n");
        goto end;
    }

    /*
     * Set the endpoints for the filter graph. The filter_graph will
     * be linked to the graph described by filters_descr.
     */

     /*
      * The buffer source output must be connected to the input pad of
      * the first filter described by filters_descr; since the first
      * filter input label is not specified, it is set to "in" by
      * default.
      */
    outputs->name = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx = 0;
    outputs->next = NULL;

    /*
     * The buffer sink input must be connected to the output pad of
     * the last filter described by filters_descr; since the last
     * filter output label is not specified, it is set to "out" by
     * default.
     */
    inputs->name = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx = 0;
    inputs->next = NULL;

    if ((ret = avfilter_graph_parse_ptr(filter_graph, filters_descr,
        &inputs, &outputs, NULL)) < 0)
        goto end;

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0)
        goto end;

end:
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    return ret;
}

static void display_frame(const AVFrame* frame, AVRational time_base, const char* wnd, int wait)
{
    int x, y;
    uint8_t* p0, * p;
    int64_t delay;

    if (frame->pts != AV_NOPTS_VALUE) {
        if (last_pts != AV_NOPTS_VALUE) {
            /* sleep roughly the right amount of time;
             * usleep is in microseconds, just like AV_TIME_BASE. */
            delay = av_rescale_q(frame->pts - last_pts,
                time_base, AV_TIME_BASE_Q);

            // 按正常速度播放画面需delay
            //if (delay > 100 && delay < 1000000)
            //    std::this_thread::sleep_for(std::chrono::microseconds(delay - 100));
        }
        last_pts = frame->pts;
    }

    cv::Mat img = AVFrame_to_CvMat(frame);
    cv::imshow(wnd, img);
    cv::waitKey(wait);
}

void save_frame(const AVFrame* frame, const char* path, int shot_index, int frame_index, int first_0_last_1)
{
    cv::Mat mat = AVFrame_to_CvMat(frame);

    char file[1024];
    sprintf_s(file, 1024, "%s\\shot_%06d.%s.%06d.png", path, shot_index, first_0_last_1 == 0 ? "first" : "last", frame_index);
    cv::imwrite(file, mat);

    std::cout << file << std::endl;
}

std::string create_shots_directory(const char* video_file)
{
    std::string file(video_file);
    int last = file.find_last_of('\\');

    char dir[1024];
    sprintf_s(dir, 1024, "%s\\shots", file.substr(0, last).c_str());
    ::CreateDirectory(dir, NULL);
    sprintf_s(dir, 1024, "%s\\shots\\%s", file.substr(0, last).c_str(), file.substr(last + 1).c_str());
    ::CreateDirectory(dir, NULL);
    sprintf_s(dir, 1024, "%s\\shots\\%s\\origin", file.substr(0, last).c_str(), file.substr(last + 1).c_str());
    ::CreateDirectory(dir, NULL);

    return std::string(dir);
}

int main(int argc, char** argv)
{
    int ret;
    AVPacket* packet;
    AVFrame* last_frame = NULL;
    AVFrame* frame;
    AVFrame* filt_frame;
    int frame_index = 1;
    int shot_index = 1;
    std::string dir;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s file\n", argv[0]);
        exit(1);
    }

    dir = create_shots_directory(argv[1]);

    frame = av_frame_alloc();
    filt_frame = av_frame_alloc();
    packet = av_packet_alloc();
    if (!frame || !filt_frame || !packet) {
        fprintf(stderr, "Could not allocate frame or packet\n");
        exit(1);
    }

    if ((ret = open_input_file(argv[1])) < 0)
        goto end;
    if ((ret = init_filters(filter_descr)) < 0)
        goto end;

    /* read all packets */
    while (1) {
        if ((ret = av_read_frame(fmt_ctx, packet)) < 0)
            break;

        if (packet->stream_index == video_stream_index) {
            ret = avcodec_send_packet(dec_ctx, packet);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error while sending a packet to the decoder\n");
                break;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                else if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error while receiving a frame from the decoder\n");
                    goto end;
                }

                frame->pts = frame->best_effort_timestamp;
                
                // 调试显示帧图像
                //display_frame(frame, dec_ctx->time_base, "img", 1);

                if (frame_index == 1)
                    save_frame(frame, dir.c_str(), shot_index, frame_index, 0);

                /* push the decoded frame into the filtergraph */
                if (av_buffersrc_add_frame_flags(buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error while feeding the filtergraph\n");
                    break;
                }

                /* pull filtered frames from the filtergraph */
                while (1) {
                    ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    if (ret < 0)
                        goto end;

                    // 保存上一镜头尾帧
                    save_frame(last_frame, dir.c_str(), shot_index, frame_index - 1, 1);

                    // 保存新镜头首帧
                    shot_index++;
                    save_frame(frame, dir.c_str(), shot_index, frame_index, 0);

                    // 调试显示新的分镜头图像
                    //display_frame(filt_frame, buffersink_ctx->inputs[0]->time_base, "filter", 1);

                    av_frame_unref(filt_frame);
                }

                av_frame_unref(last_frame);
                av_frame_free(&last_frame);

                last_frame = av_frame_clone(frame);
                av_frame_unref(frame);

                frame_index++;
            }
        }
        av_packet_unref(packet);
    }
end:
    avfilter_graph_free(&filter_graph);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    av_frame_free(&filt_frame);

    if (last_frame != NULL)
    {
        save_frame(last_frame, dir.c_str(), shot_index, frame_index - 1, 1); // 保存最后一副图
        av_frame_free(&last_frame);
    }

    av_packet_free(&packet);

    if (ret < 0 && ret != AVERROR_EOF) {
        fprintf(stderr, "Error occurred: %d\n", ret);
        exit(1);
    }

    exit(0);
}
