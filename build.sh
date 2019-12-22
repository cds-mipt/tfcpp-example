g++ -std=c++11 -c main.cc -I /usr/include/tensorflow/ && g++ -std=c++11 main.o -ltensorflow_cc $(pkg-config --cflags --libs opencv) -o tl_classifier
