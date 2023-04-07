#include <vector>
#include <iostream>
#include "fstream"
#pragma omce

#include "string"
#include "sstream"

#define NN_ICON_SIZE 8*8

#define DATA_PREP_UNSIG 1       // (val-min)/(max-min)
#define DATA_PREP_ZERO_AV 2     // (val-min - (max-min)/2) / (max-min)/2
#define DATA_PREP_ZERO_WAV 4    // (val-min - (WA-min) / (WA-min)

// ��������� ������ ������ � ������������ ������
struct NNIcon {
    float data_[NN_ICON_SIZE];  //������ �������� �������� ������ �� ����� ����� �� ����� ����
    unsigned value_;     //�����, ������������ � ������
};

// �����, ���������� ������ ���� ����������� ������
class NNData {
private:
    bool prepared_ = false; //������������ �� ������ ������
public:
    std::vector<NNIcon> icon_vect_; //������ � ��������

    size_t Read(const std::string& file_name); //��������� ������ ������ �� �����
    void Prepare(unsigned prep_way = DATA_PREP_ZERO_WAV); //��������������� ��������� ������ �����
};


