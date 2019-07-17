#encoding:utf-8
#seek策略
#一共2种地块，要么是沙地，要么是耕地；第一块地一定是耕地；
#1.1,当前是沙地，如果连续沙地大于等于15m($max_sand_len)，停止seek，目标点在连续沙地之前的最后一点；
#1.2,当前是沙地，如果连续沙地小于15m($max_sand_len)，继续seek；
#2.1,当前是耕地，如果已有耕地之和小于10m($min_farm_len)，继续seek；
#2.2,当前是耕地，如果已有耕地之和在10m和20m之间($min_farm_len <= x <= $max_farm_len)，停止，目标点在当前地块的尽头；
#2.3,当前是耕地，如果已有耕地之和大于20m($max_farm_len)，并且当前是第一块地块，则目标点在当前地块的第20m($max_farm_len)处。
#2.4,当前是耕地，如果已有耕地之和大于20m($max_farm_len)，并且当前不是第一块地块，并且当前地块小于等于10m($final_blow_len)，目标点在当前地块的尽头，
#2.5,当前是耕地，如果已有耕地之和大于20m($max_farm_len)，并且当前不是第一块地块，并且当前地块大于10m($final_blow_len)，目标点在当前地块的第10m($final_blow_len)处。

#hyper parameter
input_lot_number = 6
max_sand_len = 15
max_farm_len = 20
min_farm_len = 10
final_blow_len = 10
sand_type = -1
farm_type = 1


#sequence strategy just like human
#[(1, 10), (1,10), (-1, 10), (-1, 10), (-1, 10), (-1, 10)]
def getPlantPoint(lot_list):
    if len(lot_list) < input_lot_number:
        raise Exception("Illegal input. input should be a list with len {} whereas you provided is {}".format(input_lot_number, len(lot_list)))
    if len(lot_list) > input_lot_number:
        lot_list = lot_list[:input_lot_number]
    
    #the 1st lot
    plant_point = lot_list[0][1]
    #which plot
    plant_cls1 = 0
    #whether cropped, 1 means cropped.
    plant_cls2 = 0

    assert lot_list[0][0] == farm_type, "1st lot type should be farm."

    continuous_sand = []
    total_farm = []
    total_lot = []
    for i in range(input_lot_number):
        lot_type, lot_len = lot_list[i]
        total_lot.append(lot_len)
        plant_cls1 = i

        #current is sand
        if lot_type == sand_type:
            continuous_sand.append(lot_len)
            #condition #1.1
            if sum(continuous_sand) >= max_sand_len:
                break
            #condition #1.2
            continue
        
        #current is farm
        del continuous_sand[:]
        total_farm.append(lot_len)
        #condition #2.1
        if sum(total_farm) < min_farm_len:
            continue
        #condition #2.2
        if min_farm_len <= sum(total_farm) <= max_farm_len:
            plant_point = sum(total_lot)
            plant_cls2 = 0
            break
        #condition #2.3
        if len(total_farm) == 1:
            plant_point = max_farm_len
            plant_cls2 = 1
            break
        #condition #2.4
        if lot_len <= final_blow_len:
            plant_point = sum(total_lot)
            plant_cls2 = 0
            break
        #condition #2.5
        plant_point = sum(total_lot[:-1]) + final_blow_len
        plant_cls2 = 1
    
    return (plant_point, plant_cls1, plant_cls2)

        

#用来产生相关机器学习模型的数据
if __name__ == '__main__':
    example1 = [(1, 2), (1,3), (-1, 10), (1, 4), (-1, 10), (1, 20)]
    print(getPlantPoint(example1))
        

        


    
    
    

