#include "detection.hpp"

Detection::Detection(const int classId, const float score, int *locations) : classId(classId), score(score)
{
    this->bbox.xMin = locations[0];
    this->bbox.yMin = locations[1];
    this->bbox.xMax = locations[2];
    this->bbox.yMax = locations[3];
}

int Detection::getClassId(void) { return this->classId; }
float Detection::getScore(void) { return this->score; }
BoundingBox Detection::getBoundingBox(void) { return this->bbox; }