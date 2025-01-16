#include "Configure.h"

// TODO(optional)
// add yor parameters

// system 
int imageWidth = 600;       // ��Ⱦ�ֱ���
int imageHeight = 600;

int windowWidth = 1080;     // ����Ĭ�Ϸֱ���
int windowHeight = 960;

float fontSize = 16.0f;     // �����С

bool simulating = false;    // ģ��״̬������л���ͣ


// 2d ŷ���������õ��Ĳ���
namespace Eulerian2dPara
{
    // MAC grid
    int theDim2d[2] = {100, 100};   // ����ά��
    float theCellSize2d = 0.5;
    
    // �����ͷ����������
    std::vector<SourceSmoke> source = {
        {   // ��������                      // �����ʼ�ٶ�       //�ܶ� // �¶�
            glm::ivec2(theDim2d[0]/3, 0), glm::vec2(0.0f, 1.0f), 1.0f, 1.0f
        }
    };

    bool addSolid = false;

    // renderer settings
    float contrast = 1;     // �����Աȶ�
    int drawModel = 0;
    int gridNum = theDim2d[0];

    // solver settings
    float dt = 0.005;
    float airDensity = 1.3;         // �����ܶ�
    float ambientTemp = 0.0;        // �����¶�
    float boussinesqAlpha = 500;  // Boussinesq Force ��ʽ�е� alpha ����
    float boussinesqBeta = 2500.0;  // Boussinesq Force ��ʽ�е� beta ����
}

namespace Eulerian3dPara
{
    // MAC grid
    int theDim3d[3] = {12, 36, 36}; // ��ȷ�� x <= y = z
    float theCellSize3d = 0.5;
    std::vector<SourceSmoke> source = {
        {glm::ivec3(theDim3d[0] / 2, theDim3d[1] / 2, 0), glm::vec3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f}
    };
    bool addSolid = true;

    // renderer settings
    float contrast = 1;
    bool oneSheet = true;
    float distanceX = 0.51;
    float distanceY = 0.51;
    float distanceZ = 0.985;
    bool xySheetsON = true;
    bool yzSheetsON = true;
    bool xzSheetsON = true;
    int drawModel = 0;
    int gridNumX = (int)((float)theDim3d[0] / theDim3d[2] * 100);
    int gridNumY = (int)((float)theDim3d[1] / theDim3d[2] * 100);
    int gridNumZ = 100;
    int xySheetsNum = 3;
    int yzSheetsNum = 3;
    int xzSheetsNum = 3;

    // solver settings
    float dt = 0.01;
    float airDensity = 1.3;
    float ambientTemp = 0.0;
    float boussinesqAlpha = 500.0;
    float boussinesqBeta = 2500.0;
}

// 2d ���������������õ��Ĳ���
namespace Lagrangian2dPara
{
    // scale: ��ά������scale��
    // ͨ������scale�Ĵ�С���Ŵ�������������С���Ӷ�����������
    float scale = 2;

    // ��ʼ���õ��������
    std::vector<FluidBlock> fluidBlocks = {
        {   // ���½�����             // ���Ͻ�����           // ���ٶ�              // ����֮��ļ��
            glm::vec2(-0.4f, -0.4f), glm::vec2(0.4f, 0.4f), glm::vec2(0.0f, 0.0f), 0.02f
        }
    };

    // solver settings
    float dt = 0.0016;
    int substep = 1;
    float maxVelocity = 10;             // ��������ٶ�
    float velocityAttenuation = 0.7;    // �������߽����ٶ�˥��ϵ��
    float eps = 1e-5;                   // һ����С�ľ��룬���ڱ߽��⣬��ֹ������������

    // particle system
    float supportRadius = 0.04;     // �ڴ˰뾶�ڵ��������ӻ�Ե�ǰ���Ӳ���Ӱ��
    float particleRadius = 0.01;    // ���ӵİ뾶
    float particleDiameter = particleRadius * 2.0;
    float gravityX = 0.0f;          // x���ϵļ��ٶ�
    float gravityY = 9.8f;          // y���ϵļ��ٶ�
    float density = 1000.0f;        // �����ܶ�
    float stiffness = 70.0f;        // �ն�
    float exponent = 7.0f;          // ѹ�����㹫ʽ��ָ��
    float viscosity = 0.03f;        // ճ��
}

namespace Lagrangian3dPara
{
    // scale
    float scale = 1.2;
    std::vector<FluidBlock> fluidBlocks = {
        {
            glm::vec3(0.05, 0.05, 0.3), glm::vec3(0.45, 0.45, 0.7), glm::vec3(0.0, 0.0, -1.0), 0.02f
        },
        {
            glm::vec3(0.45, 0.45, 0.3), glm::vec3(0.85, 0.85, 0.7), glm::vec3(0.0, 0.0, -1.0), 0.02f
        }   
    };
    
    // solver settings
    float dt = 0.002;
    int substep = 1;
    float maxVelocity = 10;
    float velocityAttenuation = 0.7;
    float eps = 1e-5;

    // particle system
    float supportRadius = 0.04;
    float particleRadius = 0.01;
    float particleDiameter = particleRadius * 2.0;
    float gravityX = 0.0f;
    float gravityY = 0.0f;
    float gravityZ = 9.8f;
    float density = 1000.0f;
    float stiffness = 20.0f;
    float exponent = 7.0f;
    float viscosity = 8e-5f;
}

// stores system's all components
std::vector<Glb::Component *> methodComponents;

std::string shaderPath = "../../../../code/resources/shaders";
std::string picturePath = "../../../../code/resources/pictures";