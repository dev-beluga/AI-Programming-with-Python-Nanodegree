def create_flowerdict(filename):
    flower_dict = {}
    with(open(filename)) as file:
         for line in file:
             letter = line.split(": ")[0].lower()
             flower = line.split(": ")[1].strip()
             flower_dict[letter] = flower
    return flower_dict

def main():
    flower_dic = create_flowerdict('flowers.txt')
    f_name = input("Enter your First [space] Last name only: ")
    first_name = f_name[0].lower()
    
    print("Name with flower: {}".format(flower_dic[first_name]))

main()