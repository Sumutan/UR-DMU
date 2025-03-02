"""
根据特征名称快速生成启动配置
"""


#UCF
def UCF_videoMAE_config(rgb_list:str):
    assert 'ucf-videoMae-' in rgb_list
    template=r"""ucf-videoMae-FeatureName:
    --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss  --use_dic_gt
    --feat-extractor videoMAE --feature-size 1280
    --rgb-list ./list/ucf-videoMae-FeatureName.list
    --test-rgb-list ./list/ucf-videoMae-test-FeatureName.list
    --exp-name ucf-videoMae-FeatureName
    """
    # print(template)
    new_feature=rgb_list.replace('ucf-videoMae-','').replace('.list','')
    config=template.replace("FeatureName",new_feature)
    return config

def TAD_videoMAE_config(rgb_list:str):
    assert 'TAD-videoMae-' in rgb_list
    template=r"""TAD-videoMae-FeatureName:
    --dataset TAD --feature-group both --fusion concat --feature-size 1280 --use_dic_gt
    --feat-extractor videoMAE --aggregate_text --extra_loss --batch-size 64
    --rgb-list list/TAD-videoMae-FeatureName.list
    --test-rgb-list list/TAD-videoMae-test-FeatureName.list
    --exp-name TAD-videoMae-FeatureName
    """
    # print(template)
    new_feature=rgb_list.replace('TAD-videoMae-','').replace('.list','')
    config=template.replace("FeatureName",new_feature)
    return config

def SHT_videoMAE_config(rgb_list:str):
    assert 'SHT-videoMae-' in rgb_list
    template=r"""SHT-videoMae-FeatureName:
    --dataset shanghai --feature-group both --fusion add --aggregate_text --extra_loss --use_dic_gt
    --feat-extractor videoMAE --feature-size 1280
    --rgb-list ./list/SHT-videoMae-FeatureName.list
    --test-rgb-list ./list/SHT-videoMae-test-FeatureName.list
    --exp-name SHT-videoMAE-FeatureName
    """
    # print(template)
    new_feature=rgb_list.replace('SHT-videoMae-','').replace('.list','')
    config=template.replace("FeatureName",new_feature)
    return config

def XD_videoMAE_config(rgb_list:str):
    assert 'violence-videoMae-' in rgb_list
    template=r"""violence-videoMae-FeatureName:
    --dataset violence --feature-group both --fusion concat --feature-size 1280 --use_dic_gt
    --feat-extractor videoMAE --aggregate_text --extra_loss 
    --rgb-list list/violence-videoMae-FeatureName.list
    --test-rgb-list list/violence-videoMae-test-FeatureName.list
    --exp-name violence-videoMae-FeatureName
    """
    # print(template)
    new_feature=rgb_list.replace('violence-videoMae-','').replace('.list','')
    config=template.replace("FeatureName",new_feature)
    return config

def buildConfigFile(rgb_lists):
    new_configs=[]
    for rgb_list in rgb_lists:
        if 'ucf' in rgb_list:
            new_config=UCF_videoMAE_config(rgb_list)
        elif 'TAD' in rgb_list:
            new_config=TAD_videoMAE_config(rgb_list)
        elif 'SHT' in rgb_list:
            new_config = SHT_videoMAE_config(rgb_list)
        elif 'violence' in rgb_list:
            new_config = XD_videoMAE_config(rgb_list)
        else:
            raise ValueError
        new_configs.append(new_config)
    print(new_configs)
    with open("config_.txt","w") as f:
        for config in new_configs:
            f.write('\n'+config)
        for config in new_configs:
            f.write('\n'+config.replace('\n','').split(":    ")[1])
        f.write('\n')
        #other seed
        for config in new_configs:
            f.write('\n'+config.replace('--dataset','--seed 228 --dataset').replace('\n','')
                    .split(":    ")[1].rstrip(" ")+"_seed228")
        f.write('\n')
        for config in new_configs:
            f.write('\n'+config.replace('--dataset','--seed 3407 --dataset').replace('\n','')
                    .split(":    ")[1].rstrip(" ")+"_seed3407")
        f.write('\n')

if __name__ == '__main__':
    #将list文件放入列表，生成config
    rgb_lists=[
        'TAD-videoMae-9-5_9-1_finetune_AISO_0.5_SP_norm_a0.05-residual.list',
    ]
    buildConfigFile(rgb_lists)



