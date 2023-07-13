//
//  OCVHungarianAlgorithm.h
//  OpenCVRecognizer
//
//  Created by Yanis Plumit on 04.07.2023.
//

// HungarianAlgorithm http://acm.mipt.ru/twiki/bin/view/Algorithms/HungarianAlgorithmCPP

#include <vector>
#include <limits>
using namespace std;

#ifndef OCVHungarianAlgorithm_h
#define OCVHungarianAlgorithm_h

typedef pair<int, int> PInt;
typedef vector<int> VInt;
typedef vector<VInt> VVInt;
typedef vector<PInt> VPInt;

extern VPInt hungarian(const VVInt &matrix);

#endif // OCVHungarianAlgorithm_h
