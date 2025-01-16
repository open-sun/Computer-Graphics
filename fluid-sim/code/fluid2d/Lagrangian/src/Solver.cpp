#include "Lagrangian/include/Solver.h"
#include "Global.h"
#include <iostream>
#include <algorithm>

namespace FluidSimulation
{

    namespace Lagrangian2d
    {
        Solver::Solver(ParticleSystem2d &ps) : mPs(ps)
        {
        }


        void Solver::solve()
        {
            // TODO
            // Solves the fluid simulation by performing some steps, which may include:
            // 1. compute density 
            // 2. compute press
            // 3. compute accleration
            // 4. update velocity and position
            // 5. check boundary
            // 6. update block id
            // ...
            // ��1�������������ܶ�
            computeDensity();

            // ��2������������ѹ��
            computePressure();

            // ��3�����������ӵļ��ٶ�
            computeAcceleration();

            // ��4�����������ӵ��ٶȺ�λ��
            updateVelocityAndPosition();

            // ��5�������߽�����
            checkBoundary();

            // ��6������������������Ϣ
            updateBlockInfo();
        }
        // ��Solver���У���Ӽ���Poly6�˺����ĳ�Ա����
       // Poly6 �˺���
        float Solver::W(float dist)
        {
            // ����������֧�ְ뾶������0
            if (dist >= mPs.supportRadius)
            {
                return 0.0f;
            }

            const float h = mPs.supportRadius;
            const float coeff = 315.0f / (64.0f *3.1415926 * pow(h, 9));  // ������

            float distSquared = dist * dist;
            return coeff * pow(h * h - distSquared, 3);
        }


        void Solver::computeDensity()
        {
            // ������������
            for (auto& particle : mPs.particles)
            {
                particle.density = 0.0f;  // ���������ܶ�
                // �����������ӣ�����ÿ�����ӵ��ܶ�
                for (const auto& neighbor : mPs.particles)
                {
                    if (&particle != &neighbor)
                    {
                        float dist = glm::length(particle.position - neighbor.position);
                        if (dist < mPs.supportRadius)
                        {
                            // ����ÿ�����ӵ�����Ϊ1��W��ƽ���˺���
                            float weight = W(dist);
                            particle.density += weight;  // �����ܶ�
                        }
                    }
                }
            }
        }
        void Solver::computePressure()
        {
            // ������������
            for (auto& particle : mPs.particles)
            {
                float density_ratio = particle.density / Lagrangian2dPara::density;

                // ȷ��ѹ�������Ϊ������������ͨ��������ѹ����
                float pressure = Lagrangian2dPara::stiffness * (pow(density_ratio, Lagrangian2dPara::exponent) - 1.0f);

                // ���ѹ��С��0, ����Ϊ0
                particle.pressure = max(pressure, 0.0f);
            }
        }
        void Solver::computeAcceleration()
        {
            for (auto& particle : mPs.particles)
            {
                glm::vec2 pressureForce(0.0f);
                glm::vec2 viscosityForce(0.0f);

                // �����������ӣ�������
                for (auto& neighbor : mPs.particles)
                {
                    if (&particle != &neighbor)
                    {
                        float dist = glm::length(particle.position - neighbor.position);
                        if (dist < mPs.supportRadius)
                        {
          
                            glm::vec2 dir = particle.position - neighbor.position;
                            float weight = W(dist);
                            pressureForce += (particle.pressure + neighbor.pressure) / (2.0f * neighbor.density) * weight * glm::normalize(dir);

                            // ճ����: viscosity * (v_i - v_j) * W
                            viscosityForce += Lagrangian2dPara::viscosity * (neighbor.velocity - particle.velocity) * weight;
                        }
                    }
                }

                // Ӧ��ѹ������ճ����������
                glm::vec2 acceleration = (pressureForce + viscosityForce) / particle.density;
                acceleration += glm::vec2(Lagrangian2dPara::gravityX, -Lagrangian2dPara::gravityY);

                // �洢����õ��ļ��ٶ�
                particle.accleration = acceleration;
            }
        }
        void Solver::updateVelocityAndPosition()
        {
            for (auto& particle : mPs.particles)
            {
                // ʹ�ü��ٶȸ����ٶ�
                particle.velocity += particle.accleration * Lagrangian2dPara::dt;

                // ��������ٶ�
                if (glm::length(particle.velocity) > Lagrangian2dPara::maxVelocity)
                {
                    particle.velocity = glm::normalize(particle.velocity) * Lagrangian2dPara::maxVelocity;
                }

                // ����λ��
                particle.position += particle.velocity * Lagrangian2dPara::dt;
            }
        }
        void Solver::checkBoundary()
        {
            for (auto& particle : mPs.particles)
            {
                // ��������Ƿ񳬳��߽粢���з���
                if (particle.position.x < mPs.lowerBound.x + Lagrangian2dPara::eps)
                {
                    particle.position.x = mPs.lowerBound.x + Lagrangian2dPara::eps;
                    particle.velocity.x *= -Lagrangian2dPara::velocityAttenuation;
                }
                if (particle.position.x > mPs.upperBound.x - Lagrangian2dPara::eps)
                {
                    particle.position.x = mPs.upperBound.x - Lagrangian2dPara::eps;
                    particle.velocity.x *= -Lagrangian2dPara::velocityAttenuation;
                }

                if (particle.position.y < mPs.lowerBound.y + Lagrangian2dPara::eps)
                {
                    particle.position.y = mPs.lowerBound.y + Lagrangian2dPara::eps;
                    particle.velocity.y *= -Lagrangian2dPara::velocityAttenuation;
                }
                if (particle.position.y > mPs.upperBound.y - Lagrangian2dPara::eps)
                {
                    particle.position.y = mPs.upperBound.y - Lagrangian2dPara::eps;
                    particle.velocity.y *= -Lagrangian2dPara::velocityAttenuation;
                }
            }
        }
        void Solver::updateBlockInfo()
        {
            mPs.updateBlockInfo();
        }






    }
}
