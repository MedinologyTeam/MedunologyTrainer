#ifndef __PLATFORM_H
#define __PLATFORM_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#include "csvparser.h"
#include "Eigen/Eigen/Dense"

#define PI 3.14159265359
using namespace Eigen;
using namespace std;

#define SYMPTOM_NUM 51
#define DISEASE_NUM 31
#define MEDICINE_NUM 36
#define HIDDEN_NUM 200
extern ofstream logger;
#endif

