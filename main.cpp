#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "Detector.h"


namespace fs = std::filesystem;

const fs::path modelPath = "model/ssdlite_mobilenet_v2.xml";
const fs::path videoPath = "test_video4.mp4";


void processFrame(cv::Mat& frame, Detector& detector, const std::vector<std::string>& classNames) {
	std::vector<Object> objects;
	detector.Detect(frame, objects);

	for (const auto& obj : objects) {
		cv::rectangle(frame, obj.box, cv::Scalar(255, 0, 255), 2);
		cv::putText(frame, cv::format("%.1f", obj.score), obj.box.tl() + cv::Point(obj.box.width - 50, 0), 0, 0.5, cv::Scalar(255, 0, 255), 1);
		cv::putText(frame, classNames[obj.label], obj.box.tl(), 0, 0.5, cv::Scalar(255, 0, 255), 1);
	}
}


int main(int argc, char* argv[]) {
	fs::path currentPath = fs::current_path();

	fs::path modelPathFull = currentPath / modelPath;
	fs::path videoPathFull = currentPath / videoPath;

	// Init Detector
	Detector detector(modelPath.string());
	detector.InitModel();

	// Video Capture
	cv::VideoCapture cap(videoPath.string());
	if (!cap.isOpened())
		return -1;

	int fpsCount = 0;
	auto start = std::chrono::steady_clock::now();

	std::vector<Object> objects;

	while (true) {
		cv::Mat frame;
		cap >> frame;

		if (frame.empty())
			break;

		cv::resize(frame, frame, cv::Size(640, 360));

		// Process Frame
		processFrame(frame, detector, classNames);

		// Cal fps
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		++fpsCount;

		if (elapsed.count() >= 1000) {
			double fps = static_cast<double>(fpsCount) / (elapsed.count() / 1000.0);

			std::cout << "FPS: " << fps << std::endl;

			fpsCount = 0;
			start = std::chrono::steady_clock::now();
		}

		cv::imshow("result ", frame);
		cv::waitKey(5);
	}

	return 0;
}
