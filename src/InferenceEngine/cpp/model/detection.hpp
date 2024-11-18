#ifndef DETECTION_HPP
#define DETECTION_HPP

struct BoundingBox
{
    unsigned int xMin; 
    unsigned int yMin; 
    unsigned int xMax; 
    unsigned int yMax;
};

class Detection
{
private:
    const int classId; 
    const float score;
    BoundingBox bbox;

public:
    Detection(const int classId, const float score, int *locations);

    int getClassId(void);
    float getScore(void);
    BoundingBox getBoundingBox(void);
};

#endif // DETECTION_HPP