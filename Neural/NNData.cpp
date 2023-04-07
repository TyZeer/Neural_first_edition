#include "NNData.h"

//--------------------------------------------
// ������ �������� ������ �� �����
//
size_t NNData::Read(const std::string& file_name) {

    std::ifstream data_file;
    data_file.open(file_name);
    if (!data_file.is_open()) { return -1; }
    //  ���� �������� �������� ������
    size_t row_count = 0;
    std::string token, line_read;

    while (!data_file.eof()) { // ���� �� ������ �� ����� �����
        data_file >> line_read;
        if (line_read.size() == 0) { return -1; }
        icon_vect_.push_back(NNIcon()); //��������� ��������� ������ � ������ ������
        std::stringstream line_stream(line_read); //������ �� ������ ����� ��� ����������������� ���������� ��������

        int token_count = 0;
        while (getline(line_stream, token, ',')) {  // ��������� ��������� �������� �� ������� � ������ token
            try {
                if (token_count == NN_ICON_SIZE) //���� �������� ����� 64, ������ ����� �������� �������� �������� ������
                    icon_vect_[row_count].value_ = stoi(token);
                else
                    icon_vect_[row_count].data_[token_count] = stoi(token); //��������� ������ ���������� �������� ������

            }
            // ������������ ����������
            catch (std::invalid_argument const& ex) {

                return -1;
            }
            catch (std::out_of_range const& ex) {

                return -1;
            }

            // �������� �� ���������� ���������� �������
            if (token_count > NN_ICON_SIZE) { return -1; }
            token_count++;
        }
        //��������, ��� ���������� ��������� �������� ������������� ����������
        if (token_count < NN_ICON_SIZE + 1) { return -1; }

        row_count++;
    }

    data_file.close();
    return row_count;
}

//-----------------------------------------------
// �������������� ������ ������ � ������� ���������� � ������� ���� prep_way ������
//
void NNData::Prepare(unsigned prep_way)
{

    if (prepared_) return; //������ ��� ������������
    float min = 0.0F, max = 0.0F;
    float wa_icon = 0.0F;
    float wa = 0.0F, fmax = 0.0F;
    if (icon_vect_.size() == 0)
    {
        //std::cout << "������ ��� ���������� ��� \n";
        return;
    }
    min = max = icon_vect_[0].data_[0];

    for (auto& icon : icon_vect_)
    {
        wa_icon = 0;
        for (int i = 0; i < NN_ICON_SIZE; i++)
        {
            min = icon.data_[i] < min ? icon.data_[i] : min; //������� ������� � �������� �������� � ������
            max = icon.data_[i] > max ? icon.data_[i] : max;
            wa_icon += icon.data_[i]; //��� ����������� ��������� ��� �������� ������
        }
        wa_icon = wa_icon / (float)(NN_ICON_SIZE); //���������������� �������� ������ ������
        wa += wa_icon / icon_vect_.size(); // ���������� � ����������������� �� ���� ������� ��������� �� ���������� ������
    }
    wa = wa - min; // �������� ����������������

    if ((prep_way & (unsigned)DATA_PREP_UNSIG) == (unsigned)DATA_PREP_UNSIG) { //�������� ������ ����� �� 0 �� 1
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {
                icon.data_[i] = (icon.data_[i] - min) / (max - min);
            }
        }
    }
    else if ((prep_way & (unsigned)DATA_PREP_ZERO_AV) == (unsigned)DATA_PREP_ZERO_AV) { // �������� �� -1 �� 1 �� ������� �������������� � ����
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {
                fmax = max - min;
                icon.data_[i] = ((icon.data_[i] - min) - fmax / 2.0F) / (fmax / 2.0F);
            }
        }
    }
    else if ((prep_way & (unsigned)DATA_PREP_ZERO_WAV) == (unsigned)DATA_PREP_ZERO_WAV) { // �������� �� -1 �� 1 �� ���������������� � ����
        float f_divider = (max - wa) > (wa - min) ? max - wa : wa - min;
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {

                icon.data_[i] = ((icon.data_[i] - min) - wa) / f_divider;
            }
        }
    }
}
