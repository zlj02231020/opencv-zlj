// ============================================================
// 第九题：棋盘格角点检测与尺寸标定
//
// 1. 图里除了棋盘格还画了一个红色圆和一个绿色矩形，这两块干扰物的
//    颜色饱和度很高，而棋盘格是纯黑白（S≈0）。所以先把饱和的像素
//    全部刷成白色，干扰物就消失了，再拿干净的灰度图去做角点检测。
// 2. findChessboardCorners 的 patternSize 是"内角点"的个数，不是方格
//    数：13x9 个方格对应 (13-1)x(9-1) = 12x8 个内角点。这张图横比竖
//    长，所以先按 12x8 试，检出的角点就是从左上到右下、一行一行排的；
//    反过来写 8x12 也能检出（棋盘格转置一下还是棋盘格），但角点顺序
//    会变成一列一列从下往上，画出来的连线是竖的，看起来像图倒了。
// 3. 检出后用 cornerSubPix 把角点精化到亚像素。
// 4. 方格尺寸：统计相邻角点的间距取平均；真实边长 5mm，
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
    gray.setTo(Scalar(255), saturated);   // 棋盘格是黑白，填白不影响它

    // ---------- 2. 角点检测 ----------
    const int SQUARES_W = 13, SQUARES_H = 9;      // 方格数（横 x 竖）
    const int INNER_W = SQUARES_W - 1;            // 内角点数 12
    const int INNER_H = SQUARES_H - 1;            // 8

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
    // 自己按行连折线，每行一个颜色（跟参考答案图的画法一致）：
    // 行号决定色相，从红到紫。这样一眼能看出 12 个角点一行、共 8 行。
    Mat out = img.clone();
    for (int r = 0; r < H; ++r) {
        Scalar col;
        Mat hsvc(1, 1, CV_8UC3, Scalar(r * 179 / max(1, H - 1), 220, 230));
        cvtColor(hsvc, hsvc, COLOR_HSV2BGR);
        Vec3b v = hsvc.at<Vec3b>(0, 0);
        col = Scalar(v[0], v[1], v[2]);
        for (int c = 0; c + 1 < W; ++c)
            line(out, corners[r * W + c], corners[r * W + c + 1], col, 2, LINE_AA);
    }
    for (const auto& p : corners)
        circle(out, p, 2, Scalar(0, 255, 255), -1);

    // 结论写在右边空白处（x>690 这一条除了红色圆点全是白的）
    int theory = INNER_W * INNER_H;
    char buf[192];
    const int TX = 692;
    int ty = 40;
    snprintf(buf, sizeof(buf), "Corners: %d", (int)corners.size());
    putText(out, buf, Point(TX, ty), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
    ty += 26;
    snprintf(buf, sizeof(buf), "(%d x %d inner)", usedPattern.width, usedPattern.height);
    putText(out, buf, Point(TX, ty), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
    ty += 26;
    snprintf(buf, sizeof(buf), "theory: %dx%d", INNER_W, INNER_H);
    putText(out, buf, Point(TX, ty), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
    ty += 36;
    snprintf(buf, sizeof(buf), "Square: %.1f px", squarePx);
    putText(out, buf, Point(TX, ty), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 200), 1, LINE_AA);
    ty += 26;
    snprintf(buf, sizeof(buf), "Scale: %.2f px/mm", pxPerMm);
    putText(out, buf, Point(TX, ty), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 200), 1, LINE_AA);

    // ---------- 6. 控制台输出 ----------
    cout << "检测到内角点: " << corners.size() << " 个" << endl;
    cout << "理论值: (" << SQUARES_W << "-1) x (" << SQUARES_H << "-1) = "
         << theory << " 个" << endl;
    cout << "角点排列: " << usedPattern.width << " 列 x " << usedPattern.height
         << " 行（每行 " << usedPattern.width << " 个）" << endl;
    cout << "方格像素尺寸: 横 " << fixed << setprecision(2) << sqW
         << " px，竖 " << sqH << " px，平均 " << squarePx << " px" << endl;
    cout << "真实格边长: " << REAL_MM << " mm" << endl;
    cout << "像素-物理比例: " << pxPerMm << " px/mm" << endl;

    string outPath = (argc > 2) ? argv[2] : "q09_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
