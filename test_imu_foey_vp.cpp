//
// Created by liukun on 18-07-10.
//原始的动态估计俯仰角的方法是将foey作为状态变量，平行车道线的的交点(vanishing points)作为其中一个观测，
//根据陀螺数据计算的旋转矩阵可以得到另外一个观测，即pitch角。
//最后的结果以俯视图的形式显示，右边第一列代表动态更新俯仰角之后的的结果，右边第二列代表采用车静止时标定的结果进行俯视投影，
//下面一行代表每一帧估计的俯仰角与静止的俯仰角差的曲线，为了方便观察俯仰角的变化，绘制了这条曲线。
#include <iostream>
#include "fstream"
#include "string"
#include "opencv2/opencv.hpp"
#include "sstream"
#include "math.h"
#include <iomanip>
#include <queue>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#define PI 3.1415926
using namespace std;
using namespace cv;
const double degree2radian = M_PI/180;
const double static_foe_y_ = 206;
int age = 0;
double tic_last_imu_ = -1000;
double gap_time = 0.05;
double scale = 100000;
Eigen::VectorXf x_; // state
Eigen::MatrixXf P_; // state uncertainty covariance
Eigen::MatrixXf F_= Eigen::MatrixXf::Identity(2, 2); // transition matrix
Eigen::MatrixXf Q_= Eigen::MatrixXf::Zero(2, 2); // transition covariance matrix

void init_kf(){
    Eigen::MatrixXf x0(2, 1), P0(2, 2);
    x0(0, 0) = static_foe_y_;
    x0(1, 0) = 0.f;
    P0(0, 0) = 100;
    P0(1, 1) = 100 / powf(0.1f, 2.f);
    x_ = x0;
    P_ = P0;
}
bool integrate_gyro(Eigen::Quaterniond q_imu_2_img_,const double timestamp_img, double &delta_pitch,
                    queue<Eigen::Vector3d> imu_buf_, queue<double> imu_ts_buf_) {
    //cout<<"imu_buf: "<<imu_buf_.size()<<endl;
    Eigen::Quaterniond delta_q_;
    delta_q_.setIdentity();
    //cout<<"imu_ts_buf: "<<imu_ts_buf_.size()<<endl;
    //cout<<imu_ts_buf_.empty()<<endl;
    //cout<<"imu_ts_buf_.front():  "<<imu_ts_buf_.front()<<endl;
    //cout<<timestamp_img<<endl;
    //cout<<(imu_ts_buf_.front() <= timestamp_img)<<endl;
    while (!imu_ts_buf_.empty() && imu_ts_buf_.front() <= timestamp_img)
    {
        if (tic_last_imu_ < 0)
            tic_last_imu_ = imu_ts_buf_.front();
        double dt = imu_ts_buf_.front() - tic_last_imu_;
        tic_last_imu_ = imu_ts_buf_.front();
        Eigen::Vector3d gyro_for_img = imu_buf_.front();
        Eigen::Quaterniond q = Eigen::Quaterniond(1, gyro_for_img(0)*dt/2, gyro_for_img(1)*dt/2, gyro_for_img(2)*dt/2);
        delta_q_*= q;
        delta_q_.normalize();

        imu_ts_buf_.pop();
        imu_buf_.pop();
    }
    delta_q_ = Eigen::Quaterniond(q_imu_2_img_.toRotationMatrix()
                                  *delta_q_.toRotationMatrix()
                                  *q_imu_2_img_.toRotationMatrix().inverse());
    delta_q_ = delta_q_.conjugate();
    double sin_phi = -2*(delta_q_.x()*delta_q_.z()+delta_q_.w()*delta_q_.y());
    double cos_phi = sqrt(1-sin_phi*sin_phi);
    double sin_theta = 2*(delta_q_.y()*delta_q_.z()-delta_q_.w()*delta_q_.x())/(cos_phi);

    delta_pitch = asin(sin_theta);
    cout<<"delta_pitch: "<<delta_pitch<<endl;
    return true;
}
vector<double> update(Mat &img, double timestamps, queue<Eigen::Vector3d> imu_buf_,
                                           queue<double> imu_ts_buf_,Eigen::Quaterniond q_imu_2_img_){
    vector<double> res;
    //observation1: vanishing point
    Mat dst, cdst;
    vector<Vec2f> left_lines;
    vector<Point> Intersection;
    vector<Vec2f> right_lines;
    Canny(img, dst, 30, 70, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec2f> lines;
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        if (10 * PI / 180 < theta && theta < 80 * PI / 180) {
            left_lines.push_back(lines[i]);
        }
        else if (100 * PI / 180 < theta && theta < 170 * PI / 180) {
            right_lines.push_back(lines[i]);
        }
    }
    size_t i = 0, j = 0;
    double x = 0;
    double y = 0;
    Canny(img, dst, 30, 70, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
    //// draw lines
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        if (10 * PI / 180 < theta && theta < 80 * PI / 180) {
            line(img, pt1, pt2, Scalar(0, 255, 0), 3, CV_AA);
        }
        else if (100 * PI / 180 < theta && theta < 170 * PI / 180) {
            line(img, pt1, pt2, Scalar(255, 0, 0), 3, CV_AA);
        }
        else {
            line(img, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
        }
    }
    for (i = 0; i < left_lines.size(); ++i) {
        for (j = 0; j < right_lines.size(); ++j) {
            float rho_l = left_lines[i][0], theta_l = left_lines[i][1];
            float rho_r = right_lines[j][0], theta_r = right_lines[j][1];
            double denom = (sin(theta_l) * cos(theta_r) - cos(theta_l) * sin(theta_r));
            x = (rho_r * sin(theta_l) - rho_l * sin(theta_r)) / denom;
            y = (rho_l * cos(theta_r) - rho_r * cos(theta_l)) / denom;

            Point pt(x, y);
            Intersection.push_back(pt);
            circle(img, pt, 5, Scalar(0, 0, 0), 5);
        }
    }
    double res_vp = Intersection.at(0).y;
    res.push_back(res_vp);
    //cout<<"res_vp: "<<res_vp<<endl;

    //observation2:IMU
    float dfoey = 0.f;
    float var_dfoey = 100;
    double dpitch_gyro = 0.0;
    integrate_gyro(q_imu_2_img_,timestamps,dpitch_gyro,imu_buf_,imu_ts_buf_);
    dfoey = dpitch_gyro * 521; //fy
    cout<<"dfoey: "<<dfoey<<endl;
    var_dfoey = std::max(100 / 100.f, powf(dfoey * 0.1f, 2.f));
    if(0 == age){
        init_kf();
        age++;
    }else {
        F_(0, 1) = gap_time;
        double varx2, varxvx, varvx2;
        varx2 = 0.25 * pow(gap_time, 4.f) * scale;
        varvx2 = pow(gap_time, 2.f) * scale;
        varxvx = 0.5 * pow(gap_time, 3.f) * scale;
        Q_(0, 0) = varx2;
        Q_(0, 1) = Q_(1, 0) = varxvx;
        Q_(1, 1) = varvx2;
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        // update stage of kalman filter
        Eigen::MatrixXf Hx, Hj, z, R;
        int dim_obs = 3;
        Hx = Eigen::MatrixXf(3, 1);
        Hj = Eigen::MatrixXf::Zero(3, 2);
        z = Eigen::MatrixXf(3, 1);
        R = Eigen::MatrixXf::Identity(3, 3);
        int cnt = 0;
        Hx(cnt, 0) = x_(0, 0);
        Hj(cnt, 0) = 1.f;
        Hj(cnt, 1) = 0.f;
        z(cnt, 0) = res_vp;
        R(cnt, cnt) = 10;
        cnt++;
        // observation from static foe
        Hx(cnt, 0) = x_(0, 0);
        Hj(cnt, 0) = 1.f;
        Hj(cnt, 1) = 0.f;
        z(cnt, 0) = static_foe_y_;
        R(cnt, cnt) = 100;
        cnt++;

        // delta of foe_y
        Hx(cnt, 0) = x_(1, 0) * gap_time;
        Hj(cnt, 0) = 0.f;
        Hj(cnt, 1) = gap_time;
        z(cnt, 0) = dfoey;
        R(cnt, cnt) = var_dfoey;
        cnt++;

        Eigen::MatrixXf Ht = Hj.transpose();
        Eigen::MatrixXf S = Hj * P_ * Ht + R;
        Eigen::MatrixXf K = P_ * Ht * S.inverse();
        Eigen::MatrixXf y = z - Hx;
        x_ = x_ + K * y;
        Eigen::MatrixXf I_KH = Eigen::MatrixXf::Identity(2, 2) - K * Hj;
        P_ = I_KH * P_;

    }
    res.push_back(x_(0,0));
    return res;
}
Eigen::MatrixXf get_img_2_bv_homography(bool use_dfoey, float range_x, float range_y, float sx, float sy) {


}
bool plotMultiCurve(std::vector<std::vector<double>>& data,int w,int h,
                    double max,cv::Mat &outputArray,int num_points,double threshold,std::vector<std::string> &str_vector)
{
    cout<<"data: "<<data.size()<<endl;
    for(int i=0;i<data.size();i++)
    {
        if(data[i].size() > num_points)
            data[i].erase(data[i].begin());
    }
    outputArray = cv::Mat(h,w,CV_8UC3,cv::Scalar(255,255,255));

    line(outputArray,cv::Point(0,h/2),cv::Point(w,h/2),cv::Scalar(0,0,0));
    int len = data[0].size();
    int space = 50;
    int delta_x = (w-2*space)/(len);
    line(outputArray,cv::Point(space,0),cv::Point(space,h),cv::Scalar(0,0,0));

    std::vector<cv::Scalar> color_table;
    color_table.push_back(cv::Scalar(255,0,0));
    color_table.push_back(cv::Scalar(0,255,0));
    color_table.push_back(cv::Scalar(0,0,255));
    color_table.push_back(cv::Scalar(255,255,0));
    color_table.push_back(cv::Scalar(0,255,255));
    color_table.push_back(cv::Scalar(255,0,255));
    for(int i=0;i<data.size();i++)
    {
        putText(outputArray,str_vector[i]+" ",cv::Point(60,10*(i+1)),
                cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,0));
        line(outputArray,cv::Point(130,10*(i+1)),cv::Point(170,10*(i+1)),color_table[i]);
        std::stringstream ss;
        ss.clear();
        ss <<std::fixed<<std::setprecision(3)<<data[i][data[i].size()-1];
        std::string str;
        ss >> str;
        cv::putText(outputArray,"Value: "+str,cv::Point(w-400,10*(i+1)),
                    cv::FONT_HERSHEY_PLAIN,1.0,color_table[i]);
    }


    int num_scale = 5;
    for(int i=0;i<num_scale;i++)
    {
        cv::Point p1(space-5,0);
        cv::Point p2(space+5,0);
        p1.y = h/2 - ((i*max/(num_scale))/max)*(h/2 - 20);
        p2.y = p1.y;
        line(outputArray,p1,p2,cv::Scalar(0,0,0));
        putText(outputArray,std::to_string(i*max/num_scale),cv::Point(0,p1.y),
                cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,0));

    }
    for(int i=0;i<num_scale;i++)
    {
        cv::Point p1(space-5,0);
        cv::Point p2(space+5,0);
        p1.y = h/2 + ((i*max/(num_scale))/max)*(h/2 - 20);
        p2.y = p1.y;
        line(outputArray,p1,p2,cv::Scalar(0,0,0));
        putText(outputArray,std::to_string(-i*max/num_scale),cv::Point(0,p1.y),
                cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,0));
    }
    cv::Point last_data;
    for(int k=0;k<data.size();k++)
    {
        for(int i=0;i<len;i++)
        {
            cv::Point right_data;
            right_data.x = space + i*delta_x;
            if(abs(data[k][i]) <= threshold)
                right_data.y = h/2 - (data[k][i]/max)*(h/2 - 20);
            else if(data[k][i] > 0)
                right_data.y = 5;
            else
                right_data.y = h-5;
            if(i>0)
            {
                line(outputArray,last_data,right_data,color_table[k],2);
            }
            last_data = right_data;
        }
    }
}
int main(int argc, char **argv) {

    if (argc < 4) {
        std::cout << "Usage: " << "image_list_file image_root_path imu_file_path" << std::endl;
        return -1;
    }
    Eigen::MatrixXf R_w2c(3,3);
    R_w2c<<-0.0100401, -0.999787, 0.0180549,
            0.121146, -0.019139, -0.99245,
            0.992584, -0.00777703, 0.121313;
    Eigen::Quaterniond q_imu_2_img_ = Eigen::Quaterniond(0.748707,0.662821,0.00343187,0.00974317);
    //cout<<"siyuanshu:"<<q_imu_2_img_.coeffs()<<endl;
    //cout<<"rotation: "<<q_imu_2_img_.toRotationMatrix()<<endl;
    Eigen::MatrixXf R_vw2vc(3, 3);
    R_vw2vc << 0, -1, 0,
               0,  0,-1,
               1,  0, 0;
    Eigen::MatrixXf R_vw2c_ = R_w2c;
    Eigen::MatrixXf R_r2v_ = R_vw2vc * R_vw2c_.transpose();
    Eigen::MatrixXf R_vw2w_ = R_w2c.transpose() * R_vw2c_;
    string image_list_file;
    string image_root_path;
    string imu_file_path;
    ifstream fin;
    image_list_file = argv[1];//"../../sample/resource/image_list2.txt";
    image_root_path = argv[2];//"/home/lucky/Desktop/VIOData/ImageL_undistorted/";
    imu_file_path = argv[3];//"/home/lucky/Desktop/VIOData/imu_data.txt";

    const int width_src = 640; //The width of image
    const int height_src = 480; //The height of image


    std::queue<Eigen::Vector3d> imu_buf_;  		// imu data buffer
    std::queue<double> imu_ts_buf_;

    //init foey filter
    //object_filter::FOEYFilter Filter(foe_filter_config);

    //read imu data
    ifstream imu_file;
    imu_file.open(imu_file_path);
    string line;
    //namedWindow("IMU",WINDOW_NORMAL);
    while(std::getline(imu_file,line))
    {
        Eigen::Vector3d omega;
        Eigen::Vector3d acc;
        double time_stamp;
        stringstream ss;
        ss.clear();
        ss << line;
        ss >> omega(0) >> omega(1) >> omega(2) >> time_stamp;
        //cout << omega(0) <<" "<<omega(1) <<" "<<omega(2) <<endl;
        omega << omega(0)*degree2radian , omega(1)*degree2radian , omega(2)*degree2radian;
        //cout << omega(0) <<" "<<omega(1) <<" "<<omega(2) <<endl;
        //cout << std::setprecision(15)<<time_stamp*0.001<<endl;
        imu_buf_.push(omega);
        imu_ts_buf_.push(time_stamp*1e-9);
    }
    cv::VideoWriter vw;
    cv::Mat img, visImg;
    float range_x = 20.f;   // from  -x_range (m) to x_range (m)
    float range_y = 100.f;  // from 0 (m) to y_range (m)
    float sx = 10.f;        // scale of x axis
    float sy = 15.f;        // scale of y axis

    cv::Size new_img_shape(int(2 * range_x * sx), int(range_y * sy));
    cv::Mat bvImg0 = cv::Mat(new_img_shape, CV_8UC3);
    cv::Mat bvImg1 = cv::Mat(new_img_shape, CV_8UC3);
    cv::Mat img_origin = cv::Mat(new_img_shape, CV_8UC3);

    string image_name;
    bool is_first = true;
    int cnt = 0;
    fin.open(image_list_file);
    vector<vector<double>> all_frames_foeys(2,vector<double>());
    while(fin >> image_name)
    {
        img = cv::imread(image_root_path+image_name);
//      cout<<"---------"<<img.size()<<endl;
        string ts_r_frame_str = image_name.substr(0,image_name.size()-4);
        double ts_r_frame_d;
        stringstream ss0;
        ss0.clear();
        ss0 << ts_r_frame_str;
        ss0 >> ts_r_frame_d;
        ts_r_frame_d *= 1e-9;
        //cout<<"ts_r_frame_d: "<<ts_r_frame_d<<endl;
        vector<double> out_y_;
        //double out_y = static_foe_y_;
        out_y_= update(img,ts_r_frame_d,imu_buf_,imu_ts_buf_,q_imu_2_img_);
        all_frames_foeys[0].push_back(out_y_[0]-static_foe_y_);
        all_frames_foeys[1].push_back(out_y_[1]-static_foe_y_);
        //cout<<"out_y-static_foe_y_: "<<out_y-static_foe_y_<<endl;
        Mat curveImg;
        vector<string> str_vector;
        str_vector.push_back("dynamamic foey substract static foey");
        str_vector.push_back("dynamamic foey substract static foey");
        char text[100];
        sprintf(text, "frame_id: %d", cnt);
        int text_pos = 50;
        cv::putText(img, text,
                    cv::Point(50, text_pos), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
//        if (out.failed_type_ == object_filter::FOE_OK) {
//            // H2:adopt static extrinsics calibration results
//            // H1:adopt dynamic foey to compute the pitch
//            cam_param.dynamic_foe_y_ = out.y_;
//            MatrixXf H2 = object_filter::get_img_2_bv_homography(cam_param, false ,range_x, range_y, sx, sy);
//            MatrixXf H1 = object_filter::get_img_2_bv_homography(cam_param, true, range_x, range_y, sx, sy);
//            cv::Mat H1_cv = cv::Mat::zeros(3,3,CV_64F);
//            cv::Mat H2_cv = cv::Mat::zeros(3,3,CV_64F);
//            for(int r = 0;r<3;r++)
//                for(int k=0;k<3;k++)
//                {
//                    H1_cv.at<double>(r,k) = H1(r,k);
//                    H2_cv.at<double>(r,k) = H2(r,k);
//                }
//            cv::warpPerspective(img, img_origin, H2_cv, new_img_shape, cv::INTER_CUBIC);
//            cv::warpPerspective(img, bvImg1, H1_cv, new_img_shape, cv::INTER_CUBIC);
//            cout << img_origin.size()<<endl;
//
//        } else {
//            bvImg1.setTo(cv::Scalar(100, 100, 100));
//
//        }
//
//        cv::resize(img, img, cv::Size(float(bvImg0.rows)/img.rows*img.cols, bvImg0.rows));
//        img.copyTo(visImg);
//        cv::hconcat(visImg, bvImg1, visImg);
//        cout << visImg.size() << img_origin.size()<<endl;
//        cv::hconcat(visImg, img_origin, visImg);
//        cv::putText(visImg,"stabilitation video",cv::Point(visImg.cols-800,30),
//                    cv::FONT_HERSHEY_PLAIN,2.0,cv::Scalar(255,0,0));
//        cv::putText(visImg,"unstabilitation video",cv::Point(visImg.cols-400,30),
//                    cv::FONT_HERSHEY_PLAIN,2.0,cv::Scalar(255,0,0));
//        Mat curveImg;
//        vector<string> str_vector;
//        str_vector.push_back("dynamamic foey substract static foey");
          plotMultiCurve(all_frames_foeys,img.cols,300,10,curveImg,300,12,str_vector);
          cv::vconcat(img,curveImg,visImg);

          if (is_first && !vw.isOpened()) {
            is_first = false;
            vw.open("video.avi", CV_FOURCC('D', 'I', 'V', 'X'), 20.0, visImg.size());
            assert(vw.isOpened());
          }
          if (vw.isOpened()) {
            vw << visImg;
          }
          cv::imshow("IMU+消失点", visImg);
          cv::waitKey(1);
          cnt++;
    }
      if (vw.isOpened())
          vw.release();

    return 0;
}
