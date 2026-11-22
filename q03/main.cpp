// ============================================================
// 第三题：表面缺陷检测
//
// 思路：
//   测试图是带竖直渐变的金属表面，所以不能用一个固定阈值做二值化
//   （亮的那头会把正常纹理也判成缺陷）。做法是先估计背景：
//     background = 大核中值滤波(gray)
//   大核会把细划痕、小斑点这些高频细节全部抹平，剩下的就是"光滑的
//   理想表面"。用 gray - background 得到的残差图里，缺陷就是突出的
//   正负峰值，与整体亮度无关。
//   1. 残差 < -阈值  -> 暗缺陷（划痕、暗斑）
//   2. 残差 > +阈值  -> 亮点
//   3. 暗缺陷再按最小外接矩形的长宽比区分：长宽比 > 5 是划痕，
//      否则是暗斑。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q03_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat g;
    gray.convertTo(g, CV_32F);

    // ---------- 1. 背景估计 ----------
    // 用中值滤波而不是高斯模糊：中值对"少数派"像素免疫，细划痕不会
    // 被平均进背景；而且划痕贴在图像最下沿时，高斯模糊会因为边界复制
    // 把背景估歪，中值不会。
    Mat bg8;
    medianBlur(gray, bg8, 31);
    Mat bg;
    bg8.convertTo(bg, CV_32F);
    Mat diff = bg - g;              // 正数 = 比背景暗

    // ---------- 2. 两张残差掩码 ----------
    Mat darkMask, brightMask;
    threshold(diff, darkMask, 12, 255, THRESH_BINARY);
    threshold(-diff, brightMask, 35, 255, THRESH_BINARY);

    darkMask.convertTo(darkMask, CV_8U);
    brightMask.convertTo(brightMask, CV_8U);

    Mat k3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    // 开运算去掉零星噪点，再闭运算把断开的划痕接上
    // 暗掩码这里只做闭运算、不做开运算：底部的划痕只有 1 个像素宽，
    // 3x3 的开运算会把它整体腐蚀掉。噪点靠后面的"面积 + 长宽比"过滤。
    morphologyEx(darkMask, darkMask, MORPH_CLOSE,
                 getStructuringElement(MORPH_ELLIPSE, Size(9, 9)));
    morphologyEx(brightMask, brightMask, MORPH_OPEN, k3);

    // 暗斑周围在残差图上会出现一圈"亮边"（中值滤波把中心压暗后，
    // 边缘反而比估计背景亮）。这些亮边不是真缺陷，用膨胀后的暗掩码
    // 把它们减掉。
    Mat darkDil;
    dilate(darkMask, darkDil, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));
    subtract(brightMask, darkDil, brightMask);

    Mat out = img.clone();
    int nScratch = 0, nDark = 0, nBright = 0, idx = 0;

    auto drawBox = [&](const RotatedRect& rr, const Scalar& color, const string& label) {
        Point2f pts[4];
        rr.points(pts);
        for (int j = 0; j < 4; ++j)
            line(out, pts[j], pts[(j + 1) % 4], color, 1, LINE_AA);
        putText(out, label, pts[1] + Point2f(3, -4), FONT_HERSHEY_SIMPLEX, 0.45,
                Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, label, pts[1] + Point2f(3, -4), FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, LINE_AA);
    };

    // ---------- 3. 暗缺陷：划痕 vs 暗斑 ----------
    vector<vector<Point>> contours;
    findContours(darkMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& c : contours) {
        // 注意：不能只用 contourArea 过滤。1 个像素宽的细线，contourArea
        // 算出来接近 0（多边形面积），会被误当成噪点丢掉。改用外接矩形面积。
        Rect br = boundingRect(c);
        if (br.width * br.height < 60) continue;

        RotatedRect rr = minAreaRect(c);
        double w = max(rr.size.width, rr.size.height);
        double h = min(rr.size.width, rr.size.height);
        double aspect = (h > 1e-6) ? w / h : 99.0;

        // 划痕：极细长。暗斑：接近团块
        if (aspect > 5.0) {
            ++nScratch;
            drawBox(rr, Scalar(0, 0, 255), "D" + to_string(++idx));
            // 在轮廓上再描一圈细线，长划痕看得更清楚
            drawContours(out, vector<vector<Point>>{c}, 0, Scalar(255, 0, 255), 1);
        } else {
            ++nDark;
            drawBox(rr, Scalar(0, 0, 255), "D" + to_string(++idx));
        }
    }

    // ---------- 4. 亮点 ----------
    vector<vector<Point>> bc;
    findContours(brightMask, bc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& c : bc) {
        if (contourArea(c) < 25) continue;
        ++nBright;
        drawBox(minAreaRect(c), Scalar(0, 0, 255), "D" + to_string(++idx));
    }

    // ---------- 5. 输出 ----------
    cout << "划痕: " << nScratch << " 条" << endl;
    cout << "暗斑: " << nDark << " 处" << endl;
    cout << "亮点: " << nBright << " 处" << endl;
    cout << "缺陷总数: " << nScratch + nDark + nBright << " 个" << endl;

    char head[128];
    snprintf(head, sizeof(head), "Scratch:%d  Dark:%d  Bright:%d",
             nScratch, nDark, nBright);
    putText(out, head, Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 255, 255), 2, LINE_AA);

    string outPath = (argc > 2) ? argv[2] : "q03_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
