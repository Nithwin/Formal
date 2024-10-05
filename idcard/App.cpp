
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture video(0); // Open default camera (webcam)
    if (!video.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cv::dnn::Net model = cv::dnn::readNetFromDarknet("./best.pt", "best.weights");
    if (model.empty()) {
        std::cerr << "Error: Could not load YOLOv5 model." << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (video.isOpened()) {
        video >> frame;
        if (frame.empty())
            break;

        model.setInput(cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 640), cv::Scalar(), true, false));
        cv::Mat result = model.forward();

        // Process result similar to Python code
        // ...

        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}
