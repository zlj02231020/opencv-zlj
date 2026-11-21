// ============================================================
// 第九题：棋盘格角点检测与尺寸标定
//
// 1. 图里除了棋盘格还画了一个红色圆和一个绿色矩形，这两块干扰物的
//    颜色饱和度很高，而棋盘格是纯黑白（S≈0）。所以先把饱和的像素
//    全部刷成白色，干扰物就消失了，再拿干净的灰度图去做角点检测。
// 2. findChessboardCorners 的 patternSize 是"内角点"的个数，不是方格
//    数：9x13 个方格对应 (9-1)x(13-1) = 8x12 个内角点。这两个参数
//    写反了也能检出（图基本对称），所以两个顺序都试一次。
// 3. 检出后用 cornerSubPix 把角点精化到亚像素。
// 4. 方格尺寸：统计同一行相邻角点的间距取平均；真实边长 5mm，
//    像素/毫米 = 格宽(px) / 5。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q09_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 去掉彩色干扰物 ----------
    Mat hsv, gray;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat saturated;
    inRange(hsv, Scalar(0, 70, 70), Scalar(180, 255, 255), saturated);
    // 干扰物区域直接填白：棋盘格是黑白两色，填白不影响它
    gray.setTo(Scalar(255), saturated);

    // ---------- 2. 角点检测 ----------
    const int SQUARES_W = 9, SQUARES_H = 13;      // 方格数
    const int INNER_W = SQUARES_W - 1;            // 内角点数
    const int INNER_H = SQUARES_H - 1;

    vector<Point2f> corners;
    bool found = false;
    Size usedPattern;

    Size patterns[2] = {Size(INNER_W, INNER_H), Size(INNER_H, INNER_W)};
    for (auto& ps : patterns) {
        found = findChessboardCorners(
            gray, ps, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        if (found) { usedPattern = ps; break; }
        corners.clear();
    }

    if (!found) {
        cerr << "没有检测到棋盘格角点" << endl;
        return 2;
    }

    // ---------- 3. 亚像素精化 ----------
    cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                 TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01));

    // ---------- 4. 方格像素尺寸 ----------
    // 同一行里相邻两个角点的距离就是格宽；把横、竖两个方向都统计进来
    double sumH = 0, sumV = 0;
    int cntH = 0, cntV = 0;
    int W = usedPattern.width, H = usedPattern.height;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            int i = r * W + c;
            if (c + 1 < W) { sumH += norm(corners[i + 1] - corners[i]); ++cntH; }
            if (r + 1 < H) { sumV += norm(corners[i + W] - corners[i]); ++cntV; }
        }
    double sqW = cntH ? sumH / cntH : 0;
    double sqH = cntV ? sumV / cntV : 0;
    double squarePx = 0.5 * (sqW + sqH);

    const double REAL_MM = 5.0;                 // 每格真实边长 5mm
    double pxPerMm = squarePx / REAL_MM;

    // ---------- 5. 画图输出 ----------
    Mat out = img.clone();
    drawChessboardCorners(out, usedPattern, corners, found);

    // 角点再标一遍小圆，看得更清楚
    for (const auto& p : corners)
        circle(out, p, 2, Scalar(0, 255, 255), -1);

    // 理论值：(9-1)*(13-1)
    int theory = INNER_W * INNER_H;
    char buf[192];
    snprintf(buf, sizeof(buf), "Corners: %d (theory %dx%d = %d)",
             (int)corners.size(), INNER_W, INNER_H, theory);
    putText(out, buf, Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.65,
            Scalar(0, 0, 0), 3, LINE_AA);
    putText(out, buf, Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.65,
            Scalar(0, 0, 255), 1, LINE_AA);

    snprintf(buf, sizeof(buf), "Square: %.1f px = %.1f mm -> %.2f px/mm",
             squarePx, REAL_MM, pxPerMm);
    putText(out, buf, Point(10, 58), FONT_HERSHEY_SIMPLEX, 0.65,
            Scalar(0, 0, 0), 3, LINE_AA);
    putText(out, buf, Point(10, 58), FONT_HERSHEY_SIMPLEX, 0.65,
            Scalar(0, 0, 255), 1, LINE_AA);

    // 相邻角点之间连一条线，让"一格 50 像素"能直接看出来
    if (corners.size() > 1) {
        line(out, corners[0], corners[1], Scalar(0, 200, 0), 1, LINE_AA);
    }

    // ---------- 6. 控制台输出 ----------
    cout << "检测到内角点: " << corners.size() << " 个" << endl;
    cout << "理论值: (" << SQUARES_W << "-1) x (" << SQUARES_H << "-1) = "
         << theory << " 个" << endl;
    cout << "方格像素尺寸: 横 " << fixed << setprecision(2) << sqW
         << " px，竖 " << sqH << " px，平均 " << squarePx << " px" << endl;
    cout << "真实格边长: " << REAL_MM << " mm" << endl;
    cout << "像素-物理比例: " << pxPerMm << " px/mm" << endl;

    string outPath = (argc > 2) ? argv[2] : "q09_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
