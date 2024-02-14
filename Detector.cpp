#include "Detector.h"




// 모델 초기화
void Detector::InitModel() {
	ov::Core core;
	std::shared_ptr<ov::Model> model = core.read_model(m_strPath);

	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
	model = ppp.build();

	m_compile_model = core.compile_model(model, "CPU");
	m_infer = m_compile_model.create_infer_request();

	m_input_shape = m_compile_model.inputs()[0].get_shape();
	m_input_type = m_compile_model.input().get_element_type();
}


// 객체 감지 함수
void Detector::Detect(const cv::Mat& frame, std::vector<Object>& objs) {
	size_t height = m_input_shape[1];
	size_t width = m_input_shape[2];

	int frame_height = frame.rows;
	int frame_width = frame.cols;

	cv::Mat resize_frame;
	cv::resize(frame, resize_frame, cv::Size(width, height));

	float* input_data = (float*)resize_frame.data;
	ov::Tensor input_tensor = ov::Tensor(m_input_type, m_input_shape, input_data);
	m_infer.set_input_tensor(input_tensor);

	// 추론
	m_infer.infer();

	PostProcess(frame_width, frame_height, objs);
}


// 추론 데이터 후처리
void Detector::PostProcess(int width, int height, std::vector<Object>& objs) {
	auto output_tensor = m_infer.get_output_tensor();
	auto output_shape = output_tensor.get_shape();

	float* detections = output_tensor.data<float>();

	cv::Mat det_output(output_shape[2], output_shape[3], CV_32F, detections);

	std::vector<cv::Rect> boxes;
	std::vector<int> labels;
	std::vector<float> scores;

	for (int i = 0; i < det_output.rows; ++i) {
		float label = det_output.at<float>(i, 1);
		float score = det_output.at<float>(i, 2);
		float xmin = det_output.at<float>(i, 3);
		float ymin = det_output.at<float>(i, 4);
		float xmax = det_output.at<float>(i, 5);
		float ymax = det_output.at<float>(i, 6);

		boxes.push_back(cv::Rect_<float>(xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height));
		labels.push_back(int(label));
		scores.push_back(score);
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, scores, 0.6, 0.6, nms_result);

	objs.clear();
	for (int index : nms_result) {
		Object obj;
		obj.box = boxes[index];
		obj.label = labels[index];
		obj.score = scores[index];

		objs.push_back(obj);
	}
}