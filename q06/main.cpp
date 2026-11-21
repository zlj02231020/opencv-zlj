// ============================================================
// 第六题：颗粒计数与面积分析
//
// 难点在粘连：直接 findContours 会把粘在一起的几个颗粒算成一个。
// 实测这张图 24 个连通域里，最大的一个有 1.2 万像素，明显是三个
// 颗粒摞在一起，所以必须做分割。
//
// 做法：
//   1. 颗粒颜色饱和、背景纯白，用 HSV 的 S 通道阈值拿到前景掩码。
//   2. 距离变换：每个前景像素的值 = 到最近背景的距离。单个颗粒的
//      最大值就在圆心，值约等于半径。
//   3. 判断一个连通域里到底有几个颗粒：
//          比值 = 连通域面积 / (π * 最大距离²)
//      单个圆这个比值约等于 1；两个粘在一起的圆是 1.5~2；三个是 2.5~3。
//      四舍五入就得到颗粒个数 k。为什么不用"数距离变换的局部极大"？
//      实测一个颗粒的圆顶是块平台，会检出好几个只差几个像素的假峰，
//      而不规则颗粒又会在边缘冒出假峰，数量很不稳（实测 33~37 个，
//      比真值多出一截）。
//   4. 一个连通域里 k > 1 时，用 k-means 把它的像素分成 k 簇，簇心
//      当分水岭的种子；k = 1 就直接取距离最大的那个点。
//   5. 以种子为标记做 watershed，沾在一起的颗粒沿"脖子"被切开。
//      分水岭会在边界上留一圈 -1，用邻域最大值迭代扩散补上，否则
//      每个颗粒的面积都会少一圈。
//   6. 面积 < 200 px² 的当噪声丢掉，统计数量与面积分布。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q06_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 前景掩码 ----------
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsv, Scalar(0, 60, 40), Scalar(180, 255, 255), mask);

    Mat k3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, k3);
    morphologyEx(mask, mask, MORPH_CLOSE, k3);

    // ---------- 2. 距离变换 ----------
    Mat dist;
    distanceTransform(mask, dist, DIST_L2, 5);

    // ---------- 3. 每个连通域布种子 ----------
    Mat blobLbl, stats, cents;
    int nBlob = connectedComponentsWithStats(mask, blobLbl, stats, cents);

    // 标记图约定：1 = 确定的背景，2 以上 = 种子，0 = 前景里等待分配的像素
    Mat markers(mask.size(), CV_32S, Scalar(1));
    markers.setTo(0, mask);

    int nextSeed = 2;
    for (int i = 1; i < nBlob; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 200) continue;                 // 一开始就滤掉碎屑

        Mat one = (blobLbl == i);

        double dmin, dmax;
        minMaxLoc(dist, &dmin, &dmax, NULL, NULL, one);
        if (dmax < 2.0) continue;

        // 该连通域里大概有几个颗粒
        double ratio = area / (CV_PI * dmax * dmax);
        int k = (int)lround(ratio);
        if (k < 1) k = 1;
        if (ratio < 1.30) k = 1;                  // 单个圆，比值 1.0~1.1
        if (k > 6) k = 6;                         // 兜底，防止异常形状爆数量

        if (k == 1) {
            // 单个颗粒：距离最大处就是圆心
            Point peak;
            minMaxLoc(dist, &dmin, &dmax, NULL, &peak, one);
            circle(markers, peak, 1, Scalar(nextSeed++), FILLED);
        } else {
            // 粘连：把该连通域的像素聚成 k 簇，簇心当种子
            vector<Point> pts;
            for (int y = 0; y < one.rows; ++y)
                for (int x = 0; x < one.cols; ++x)
                    if (one.at<uchar>(y, x)) pts.push_back(Point(x, y));

            Mat data((int)pts.size(), 2, CV_32F);
            for (int p = 0; p < (int)pts.size(); ++p) {
                data.at<float>(p, 0) = (float)pts[p].x;
                data.at<float>(p, 1) = (float)pts[p].y;
            }
            Mat cl, ctr;
            kmeans(data, k, cl,
                   TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.5),
                   3, KMEANS_PP_CENTERS, ctr);
            for (int c = 0; c < k; ++c) {
                Point p(cvRound(ctr.at<float>(c, 0)), cvRound(ctr.at<float>(c, 1)));
                // 簇心偶尔落在掩码外（凹形颗粒），挪到最近的掩码像素上
                if (!mask.at<uchar>(p)) {
                    double best = 1e18;
                    Point bestP = p;
                    for (const auto& q : pts) {
                        double d = norm(q - p);
                        if (d < best) { best = d; bestP = q; }
                    }
                    p = bestP;
                }
                circle(markers, p, 1, Scalar(nextSeed++), FILLED);
            }
        }
    }

    // ---------- 4. 分水岭 ----------
    // 先用中值滤波压掉颗粒内部的渐变纹理。直接在原图上做分水岭，
    // 内部的小起伏会被当成山脊，分水岭线会在颗粒里乱长。
    Mat smooth;
    medianBlur(img, smooth, 7);
    watershed(smooth, markers);

    // 分水岭线（-1）和被背景标记抢走的前景像素，按最近邻并入粒子
    for (int it = 0; it < 80; ++it) {
        Mat bad = (markers <= 1) & (mask > 0);
        if (countNonZero(bad) == 0) break;
        Mat f;
        markers.convertTo(f, CV_32F);
        Mat d;
        dilate(f, d, Mat::ones(3, 3, CV_8U));
        Mat d32;
        d.convertTo(d32, CV_32S);
        d32.copyTo(markers, bad);
    }

    // ---------- 5. 统计每个颗粒 ----------
    vector<int> areas;
    Mat out = img.clone();
    for (int lb = 2; lb < nextSeed; ++lb) {
        Mat one = (markers == lb) & mask;
        int area = countNonZero(one);
        if (area < 200) continue;                 // 面积过小当噪声

        areas.push_back(area);
        vector<vector<Point>> cc;
        findContours(one, cc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (cc.empty()) continue;
        drawContours(out, cc, -1, Scalar(0, 200, 0), 1);

        Moments m = moments(one, true);
        if (m.m00 > 0) {
            Point c((int)(m.m10 / m.m00), (int)(m.m01 / m.m00));
            string num = to_string(areas.size());
            putText(out, num, c + Point(-7, 5), FONT_HERSHEY_SIMPLEX, 0.35,
                    Scalar(255, 255, 255), 3, LINE_AA);
            putText(out, num, c + Point(-7, 5), FONT_HERSHEY_SIMPLEX, 0.35,
                    Scalar(0, 0, 220), 1, LINE_AA);
        }
    }

    if (areas.empty()) {
        cerr << "没有检测到有效颗粒" << endl;
        return 2;
    }

    // ---------- 6. 输出 ----------
    long long sum = 0;
    int amin = areas[0], amax = areas[0];
    for (int a : areas) {
        sum += a;
        amin = min(amin, a);
        amax = max(amax, a);
    }

    cout << "有效颗粒数: " << areas.size() << " 个" << endl;
    cout << "最小面积: " << amin << " px²" << endl;
    cout << "最大面积: " << amax << " px²" << endl;
    cout << "平均面积: " << (int)(sum / (long long)areas.size()) << " px²" << endl;
    cout << "总面积: " << sum << " px²" << endl;

    char head[160];
    snprintf(head, sizeof(head), "Valid Particles: %d   area %d~%d px2",
             (int)areas.size(), amin, amax);
    putText(out, head, Point(10, 32), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 0, 220), 2, LINE_AA);

    string outPath = (argc > 2) ? argv[2] : "q06_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
