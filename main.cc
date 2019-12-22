#include "tensorflow/core/public/session.h"
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void init(tensorflow::Session* &session) 
{
  	std::cout << "start init" << std::endl;
	tensorflow::GraphDef graph_def;

	tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		std::cerr << "tf error: " << status.ToString() << "\n";
	}

	// Читаем граф
	status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "models/frozen_model.pb", &graph_def);
	if (!status.ok()) {
		std::cerr << "tf error: " << status.ToString() << "\n";
	}

	// Добавляем граф в сессию TensorFlow
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << "tf error: " << status.ToString() << "\n";
	}
  	std::cout << "end init" << std::endl;
}

void infer(tensorflow::Session* session, cv::Mat &image, std::vector<float> &prob)
{
	std::cout << "start infer" << std::endl;
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 96, 32, 3}));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>();

	cv::Mat dst;
	cv::resize(image, dst, cv::Size(32, 96));

	for (int y = 0; y < dst.rows; y++) {
		for (int x = 0; x < dst.cols; x++) {
			cv::Vec3b pixel = dst.at<cv::Vec3b>(y, x);

			input_tensor_mapped(0, y, x, 0) = static_cast<float>(pixel.val[2]) / 255; //R
			input_tensor_mapped(0, y, x, 1) = static_cast<float>(pixel.val[1]) / 255; //G
			input_tensor_mapped(0, y, x, 2) = static_cast<float>(pixel.val[0]) / 255; //B
		}
	}

	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{ "input_1", input_tensor }};

	std::vector<tensorflow::Tensor> output_tensors;

	auto status = session->Run(
		inputs, 
		{"fc5/Softmax"}, 
		{}, 
		&output_tensors
	);

	if (!status.ok()) {
		std::cerr << "tf error: " << status.ToString() << "\n";
	}

	for (int i = 0; i < 4; ++i) {
		prob[i] = output_tensors[0].matrix<float>()(0, i);
	}

	std::cout << "end infer" << std::endl;
}

void print_vector(std::vector<float> &x)
{
	for (float value : x) {
		printf("%.5f ", value);
	}
}

int main()
{
	std::vector<std::string> image_names = {"black.jpg", "red.jpg", "yellow.jpg", "green.jpg"};

	tensorflow::Session* session;
	init(session);

	cv::Mat image;
	std::vector<float> proba(4, 0);

	for (std::string image_name : image_names) {
		image = cv::imread("images/" + image_name, cv::IMREAD_COLOR);
		infer(session, image, proba);

		std::cout << image_name << " ";
		print_vector(proba);
		std::cout << std::endl;
	}

	return 0;
}
