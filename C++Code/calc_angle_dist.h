//
// Created by zerbeln on 1/13/22.
//

#ifndef C__CODE_CALC_ANGLE_DIST_H
#define C__CODE_CALC_ANGLE_DIST_H

using namespace std;

double get_angle_dist(double x, double y, double tx, double ty){
    double vx, vy, angle, dist;

    vx = x - tx;
    vy = y - ty;

    angle = atan2(vy, vx) * (180.0 / M_PI);
    while (angle < 0) {
        angle += 360.0;
    }
    while (angle > 360.0) {
        angle -= 360.0;
    }

    dist = (vx * vx) + (vy * vy);
    if (dist < delta_min) {
        dist = delta_min;
    }

    return angle, dist;
}

#endif //C__CODE_CALC_ANGLE_DIST_H
