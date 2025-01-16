#pragma once
#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <glad/glad.h>
#include <glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace Glb {
    class Camera {
    public:

        static Camera& getInstance() {
            static Camera instance;
            return instance;
        }

        void ProcessMove(glm::vec2 offset);                 //��������ƶ�
        void ProcessRotate(glm::vec2 offset);               //���������ת
        void ProcessScale(float offset);                    //�����������

        glm::mat4 GetView();                                //��ȡ��ͼ����
        glm::mat4 GetProjection();                          //��ȡͶӰ����
        glm::vec3 GetUp();                                  //��ȡ����Ϸ�����
        glm::vec3 GetRight();                               //��ȡ����ҷ�����
        glm::vec3 GetFront();                               //��ȡ���ǰ������
        glm::vec3 GetPosition();

        void UpdateView();                                  //������ͼ����

    private:
        Camera() {
            
            mSensitiveYaw = 0.1;            
            mSensitivePitch = 0.1;   
            mSensitiveX = 0.001;                    
            mSensitiveY = 0.001;                     
            mSensitiveFront = 0.05;

            mYaw = 13.0f;
            mPitch = 22.5f;

            mWorldUp = glm::vec3(0.0, 0.0, 1.0);
            mPosition = glm::vec3(2.0f, 1.0f, 1.178f);
            UpdateView();

            aspect = 1.0f;
            nearPlane = 0.1f;
            farPlane = 100.0f;
            fovyDeg = 60.0f;
        };

        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;

    public:
        float_t mYaw;                                     //���ƫ����
        float_t mPitch;                                   //���������

        float_t mSensitiveYaw = 0.1;                        //���ƫ����������
        float_t mSensitivePitch = 0.1;                      //���������������
        float_t mSensitiveX = 0.001;                        //����ƶ�������
        float_t mSensitiveY = 0.001;                        //����ƶ�������
        float_t mSensitiveFront = 0.05;                     //�������������

        glm::vec3 mPosition;                                //���λ��
        
        glm::vec3 mRight;                                   //����ҷ�������
        glm::vec3 mUp;                                      //����Ϸ�������
        glm::vec3 mFront;                                   //���ǰ��������
        glm::vec3 mWorldUp;                                 //�Ϸ�������


        float aspect;
        float nearPlane;
        float farPlane;
        float fovyDeg;

    };
}


#endif