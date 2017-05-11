#include <chrono>
#include <iostream>
#include <thread>

#include <math.h>
#include <unistd.h>
#include <limits.h>

#include <opencv/cv.hpp>

template <typename T>
void work_float(int width, int height, int iterations) {
	std::vector<std::vector<cv::Point>> contours;
	T frame(width, height, CV_32FC1);
	T output(width, height, CV_32FC1);

	T mean = T::zeros(1, 1, CV_32FC1);
	T sigma = T::ones(1, 1, CV_32FC1);
	cv::randn(frame, mean, sigma);

	for (auto i = 0; i < iterations; i++) {
		cv::moments(frame);
		cv::morphologyEx(frame, output, cv::MORPH_OPEN, cv::MORPH_OPEN);
		cv::matchShapes(frame, output, 1, 0.0);
	}
}

template <typename T>
void work(int depth, int width, int height, int iterations) {
	std::vector<std::vector<cv::Point>> contours;
	T frame(width, height, depth);
	T output(width, height, depth);

	switch (depth) {
		case CV_16UC1:
			cv::randu(frame, 0, USHRT_MAX);
			break;
		case CV_8UC1:
			cv::randu(frame, 0, UCHAR_MAX);
			break;
	}

	for (auto i = 0; i < iterations; i++) {
		cv::moments(frame);
		cv::morphologyEx(frame, output, cv::MORPH_OPEN, cv::MORPH_OPEN);
		cv::matchShapes(frame, output, 1, 0.0);
		if (CV_8UC1)
			cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	}
}

void usage(void) {
	std::cout << "Usage: fakealgo -d <bit depth> -w <width> -h <height> -f <num frames>" << std::endl;
	exit(EXIT_FAILURE);
}

void stats(std::vector<int>& samples) {
	double mean = 0.0;
	double variance = 0.0;
	double min = DBL_MAX;
	double max = 0.0;
	for (std::vector<int>::iterator it = samples.begin(); it != samples.end(); ++it) {
		if (*it < min) {
			min = *it;
		}

		if (*it > max) {
			max = *it;
		}

		mean += *it;
	}
	mean = mean / samples.size();

	for (std::vector<int>::iterator it = samples.begin(); it != samples.end(); ++it) {
		variance += (*it - mean) * (*it - mean);
	}
	variance = variance / samples.size();

	std::cout << "-----------------------------------------" << std::endl;
	std::cout << "Processing time per frame in milliseconds" << std::endl;
	std::cout << "-----------------------------------------" << std::endl;
	std::cout << "frames: " << samples.size() << std::endl;
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;
	std::cout << "mean: " << mean << std::endl;
	std::cout << "variance: " << variance << std::endl;
	std::cout << "standard deviation: " << sqrt(variance) << std::endl;
}

int main(int argc, char *argv[]) {
	int opt;
	int depth = 8; // 8bit depth
	int width = 160; // Default width of a frame in px.
	int height = 120; // Default height of a frame in px.
	int frames = 900; // 30 seconds of data at 30 fps.
	int iterations = 100; // Number of times to run filter chain.
	bool verbose = false;
	bool use_umat = true; // Should we use UMat or Mat?
	bool use_float = false; // Should we use floats or ints?
	std::vector<int> samples;

	while ((opt = getopt(argc, argv, "Hd:w:h:f:i:vml")) != -1) {
		switch(opt) {
			case 'H':
				usage();
				break;
			case 'd':
				depth = atoi(optarg);			
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'h':
				height = atoi(optarg);
				break;
			case 'f':
				frames = atoi(optarg);
				break;
			case 'i':
				iterations = atoi(optarg);
				break;
			case 'v':
				verbose = true;
				break;
			case 'm':
				use_umat = false;
				break;
			case 'l':
				use_float = true;
				break;
			default:
				usage();
		}
	}

	if (depth <= 0 || width <= 0 || height <= 0 || frames <= 0 || iterations <= 0) {
		usage();
	}

	// Always flush everything
	std::cout.setf(std::ios_base::unitbuf);
	std::cerr.setf(std::ios_base::unitbuf);

	for (auto i = 0; i < frames; i++) {
		auto t0 = std::chrono::high_resolution_clock::now();
		if (use_float) {
			if (use_umat)
				work_float<cv::UMat>(width, height, iterations);
			else
				work_float<cv::Mat>(width, height, iterations);
		} else {
			switch (depth) {
				case 16:
					if (use_umat)
						work<cv::UMat>(CV_16UC1, width, height, iterations);
					else
						work<cv::Mat>(CV_16UC1, width, height, iterations);
					break;
				case 8:
				default:
					if (use_umat)
						work<cv::UMat>(CV_8UC1, width, height, iterations);
					else
						work<cv::Mat>(CV_8UC1, width, height, iterations);
					break;
			}
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
		samples.push_back(dt);
		if (verbose)
			std::cerr << "time passed: " << dt << "ms" << std::endl;
		else
			std::cerr << ".";
	}

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "-----------------------------------------" << std::endl;
	std::cout << "Metadata" << std::endl;
	std::cout << "-----------------------------------------" << std::endl;
	if (use_float) {
		std::cout << "type: Float 32bit" << std::endl;
		std::cout << "note: findContours not supported" << std::endl;
	} else {
		std::cout << "type: Unsigned Int " << depth << "bit" << std::endl;
		if (depth == 16)
			std::cout << "note: findContours not supported" << std::endl;
	}
	if (use_umat) {
		std::cout << "tapi: enabled" << std::endl;
	} else {
		std::cout << "tapi: disabled" << std::endl;
	}
	std::cout << "width: " << width << "px" << std::endl;
	std::cout << "hight: " << height << "px" << std::endl;
	std::cout << "iterations: " << iterations << std::endl;
	stats(samples);
	std::cout << std::endl;

	return 0;
}
