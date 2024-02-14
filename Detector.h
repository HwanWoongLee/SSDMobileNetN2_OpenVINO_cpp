#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#pragma comment(lib, "openvino.lib")
#pragma comment(lib, "openvino_c.lib")
#pragma comment(lib, "openvino_onnx_frontend.lib")
#pragma comment(lib, "openvino_paddle_frontend.lib")
#pragma comment(lib, "openvino_pytorch_frontend.lib")
#pragma comment(lib, "openvino_tensorflow_frontend.lib")
#pragma comment(lib, "openvino_tensorflow_lite_frontend.lib")
#pragma comment(lib, "tbb.lib")
#pragma comment(lib, "tbb12.lib")
#pragma comment(lib, "tbbbind_2_5.lib")
#pragma comment(lib, "tbbmalloc.lib")
#pragma comment(lib, "tbbmalloc_proxy.lib")
#pragma comment(lib, "opencv_world480.lib")



struct Object {
	int label;
	float score;
	cv::Rect box;
};


const std::vector<std::string> classNames = {
	"background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
	"truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
	"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
	"couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
	"door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
	"teddy bear", "hair drier", "toothbrush", "hair brush"
};


class Detector {
public:
	Detector(const std::string strPath) : m_strPath(strPath) {};
	~Detector() {};

	void InitModel();
	void PostProcess(int width, int height, std::vector<Object>& objs);
	void Detect(const cv::Mat& frame, std::vector<Object>& objs);

private:
	std::string			m_strPath;

	ov::CompiledModel	m_compile_model;
	ov::InferRequest	m_infer;

	ov::Shape			m_input_shape;
	ov::element::Type	m_input_type;

};