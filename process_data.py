import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.police = dataframe['police']
        self.label = self.data.label_list
        self.max_len = max_len

    def __len__(self):
        return len(self.police)

    def __getitem__(self, index):
        input_ids = str(self.police[index])
        inputs = self.tokenizer.encode_plus(input_ids,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=True,
                                            truncation=True)
        # print('inputs',inputs)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return_data = {'ids': torch.tensor(ids, dtype=torch.long),
                       'mask': torch.tensor(mask, dtype=torch.long),
                       'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                       'label': torch.tensor(self.label[index], dtype=torch.float)}
        # print('ids',return_data['ids'].size())
        # print('mask', return_data['mask'].size())
        # print('token', return_data['token_type_ids'].size())
        # print('label',return_data['label'].size())
        return return_data


def data_loader(train_path, test_path):
    f = open(train_path, 'r', encoding='utf-8')
    Train = []
    for line in f.readlines():
        # print(line)
        line = line.split('\t')
        for i in range(0, 55):
            line[i] = int(line[i])
        # input_ids = line[2]
        Train.append(line)

    g = open(test_path, 'r', encoding='utf-8')
    Test = []
    for line in g.readlines():
        # print(line)
        line = line.split('\t')
        for i in range(0, 55):
            line[i] = int(line[i])
        # input_ids = line[2]
        Test.append(line)

    # LABELS = ['文体活动', '文物保护', '学校教育', '法律普法', '节日假日', '军队士兵警察英烈', '疫情病虫传染病', '森林林业草原', '畜牧家畜', '宠物', '津贴补贴', '农业生产', '渔业', '农村事宜', '工业生产产品', '矿物矿产', '食品药品化妆品', '高新技术', '发电供电', '供热供气制冷', '医疗医药', '健康卫生', '妇女儿童', '水源取水水利', '污染废物处理', '建筑住房', '危险物品', '灾害防治突发应急', '消防', '预算决算', '气象天气', '保险', '养老', '野生动植物', '创业就业', '人才', '社会福利扶贫救助', '扫黑除恶诈骗禁毒', '违法违规刑罚', '党建', '旅游景区', '道路交通存储运输', '车辆', '航空航天', '人口计生', '民族宗教', '丧葬', '基础设施公共设施', '资源节约环保', '港澳台外国人', '工程施工', '项目招标', '新闻广播电视电影广告', '互联网', '政务政策', '税务纳税', '金融经济', '销售经营', '公司企业社会团体', '职业执业', '计量测绘定价', '证件管理', '边境进口海关', '先进表彰', '土地土壤', '城乡规划管理', '特种作业设备', '国有资产', '社会信用']
    # LABELS = ['文体', '文物保护', '学校', '法律', '假日', '军人', '疫情', '林木', '津贴', '种植', '渔业', '工业', '矿产', '食品', '科技', '电力', '药品', '妇女', '水资源', '房屋', '灾害', '预算', '天气', '保险', '野生动植物', '就业', '人才', '扶贫', '治安', '党建', '景区', '运输', '车辆', '计划生育', '宗教', '安葬', '节约', '外国', '施工', '投标', '电视', '互联网', '纳税', '执业', '计量', '贸易', '表彰', '城乡', '国有资产', '失信']
    LABELS = ['文体', '文物保护', '学校/教育', '法律/普法', '节日/假日', '军人/军队', '疫情/病虫/传染病', '林木/林业/草原', '畜牧/家畜', '津贴/补贴', '种植', '渔业',
              '工业',
              '矿物/矿产', '食品/药品/化妆品', '科技/高新技术', '发电/供电', '供热/供气', '医疗/医药', '健康/卫生', '妇女/儿童', '水源/取水/水利', '污染/废物',
              '建筑/住房',
              '灾害/防治/突发/应急', '预算/决算', '气象/天气', '保险', '野生动物/野生植物', '创业/就业', '人才', '扶贫/社会福利', '治安', '党建', '景区',
              '道路/交通/存储/运输',
              '车辆', '计划生育/生育', '民族/宗教', '安葬', '节约/环保', '港澳台/外国人', '施工', '投标/招标', '新闻/广播/电视/电影/广告', '互联网', '税务/纳税',
              '执业/职业',
              '计量/测绘/定价', '贸易', '表彰', '土地/土壤', '城乡/城乡规划', '国有资产', '失信/社会信用']

    # df_raw = pd.DataFrame(Line,columns=['索引','文体活动', '文物保护', '学校教育', '法律普法', '节日假日', '军队士兵警察英烈', '疫情病虫传染病', '森林林业草原', '畜牧家畜', '宠物', '津贴补贴', '农业生产', '渔业', '农村事宜', '工业生产产品', '矿物矿产', '食品药品化妆品', '高新技术', '发电供电', '供热供气制冷', '医疗医药', '健康卫生', '妇女儿童', '水源取水水利', '污染废物处理', '建筑住房', '危险物品', '灾害防治突发应急', '消防', '预算决算', '气象天气', '保险', '养老', '野生动植物', '创业就业', '人才', '社会福利扶贫救助', '扫黑除恶诈骗禁毒', '违法违规刑罚', '党建', '旅游景区', '道路交通存储运输', '车辆', '航空航天', '人口计生', '民族宗教', '丧葬', '基础设施公共设施', '资源节约环保', '港澳台外国人', '工程施工', '项目招标', '新闻广播电视电影广告', '互联网', '政务政策', '税务纳税', '金融经济', '销售经营', '公司企业社会团体', '职业执业', '计量测绘定价', '证件管理', '边境进口海关', '先进表彰', '土地土壤', '城乡规划管理', '特种作业设备', '国有资产', '社会信用','police'])
    Train_df_raw = pd.DataFrame(Train, columns=['文体', '文物保护', '学校/教育', '法律/普法', '节日/假日', '军人/军队', '疫情/病虫/传染病', '林木/林业/草原', '畜牧/家畜', '津贴/补贴', '种植', '渔业', '工业', '矿物/矿产', '食品/药品/化妆品', '科技/高新技术', '发电/供电', '供热/供气', '医疗/医药', '健康/卫生', '妇女/儿童', '水源/取水/水利', '污染/废物', '建筑/住房', '灾害/防治/突发/应急', '预算/决算', '气象/天气', '保险', '野生动物/野生植物', '创业/就业', '人才', '扶贫/社会福利', '治安', '党建', '景区', '道路/交通/存储/运输', '车辆', '计划生育/生育', '民族/宗教', '安葬', '节约/环保', '港澳台/外国人', '施工', '投标/招标', '新闻/广播/电视/电影/广告', '互联网', '税务/纳税', '执业/职业', '计量/测绘/定价', '贸易', '表彰', '土地/土壤', '城乡/城乡规划', '国有资产', '失信/社会信用', 'police'])
    Train_df_raw['label_list'] = Train_df_raw[LABELS].values.tolist()
    train_dataset = Train_df_raw[['police', 'label_list']].copy()
    Test_df_raw = pd.DataFrame(Test, columns=['文体', '文物保护', '学校/教育', '法律/普法', '节日/假日', '军人/军队', '疫情/病虫/传染病', '林木/林业/草原', '畜牧/家畜', '津贴/补贴', '种植', '渔业', '工业', '矿物/矿产', '食品/药品/化妆品', '科技/高新技术', '发电/供电', '供热/供气', '医疗/医药', '健康/卫生', '妇女/儿童', '水源/取水/水利', '污染/废物', '建筑/住房', '灾害/防治/突发/应急', '预算/决算', '气象/天气', '保险', '野生动物/野生植物', '创业/就业', '人才', '扶贫/社会福利', '治安', '党建', '景区', '道路/交通/存储/运输', '车辆', '计划生育/生育', '民族/宗教', '安葬', '节约/环保', '港澳台/外国人', '施工', '投标/招标', '新闻/广播/电视/电影/广告', '互联网', '税务/纳税', '执业/职业', '计量/测绘/定价', '贸易', '表彰', '土地/土壤', '城乡/城乡规划', '国有资产', '失信/社会信用', 'police'])
    Test_df_raw['label_list'] = Test_df_raw[LABELS].values.tolist()
    valid_dataset = Test_df_raw[['police', 'label_list']].copy()
    print(f"TRAIN Dataset: {train_dataset.shape}, "
          f"TEST Dataset: {valid_dataset.shape}")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
    training_set = CustomDataset(train_dataset, tokenizer, 128)
    validation_set = CustomDataset(valid_dataset, tokenizer, 128)
    train_params = {'batch_size': 128, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': 128, 'shuffle': False, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    # for d,data in enumerate(training_loader):
    #     print('ids',data['ids'].size())
    #     print('mask', data['mask'].size())
    #     print('token', data['token_type_ids'].size())
    #     print('label',data['label'].size())
    validation_loader = DataLoader(validation_set, **test_params)
    return training_loader, validation_loader


# data_loader(train_path='P_bertPolice06_train.txt', test_path='T_bertPolice06_test.txt')
