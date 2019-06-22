import json

number_dict = {}

number_dict['02747177'] = 0
number_dict['02808440'] = 1
number_dict['02818832'] = 2
number_dict['02871439'] = 3
number_dict['02933112'] = 4
number_dict['03001627'] = 5
number_dict['03046257'] = 6
number_dict['03211117'] = 7
number_dict['03337140'] = 8
number_dict['03636649'] = 9
number_dict['03691459'] = 10
number_dict['04256520'] = 11
number_dict['04379243'] = 12

with open('number_dict.json', 'w') as fp:
    json.dump(number_dict, fp)