#include "plotter.hpp"

namespace ImagePlotter {
    static const std::vector<std::string> classes = 
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush"
    };

    void drawDetections(cv::Mat& image, Detection detection) 
    {
        BoundingBox bbox = detection.getBoundingBox();
        int classId = detection.getClassId();
        float score = detection.getScore() * 100;

        cv::rectangle(
            image, 
            cv::Point(bbox.xMin, bbox.yMin), 
            cv::Point(bbox.xMax, bbox.yMax), 
            (64, 203, 255), 
            2
        );

        std::string label = ImagePlotter::classes[classId] + ": " + std::to_string(static_cast<int>(score)) + "%";
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseline);
        int textWidth = textSize.width;

        cv::rectangle(
            image,
            cv::Point(bbox.xMin, (bbox.yMin - 15)),
            cv::Point(bbox.xMin + textWidth + 2, bbox.yMin),
            (64, 203, 255),
            cv::FILLED
        );

        cv::putText(
            image,
            label,
            cv::Point((bbox.xMin + 2), bbox.yMin),
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1
        );
    }
}