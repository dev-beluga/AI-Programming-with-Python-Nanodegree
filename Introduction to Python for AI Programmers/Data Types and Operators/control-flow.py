phone_balance = 13
wallet_balance = 1000

# print(phone_balance, wallet_balance)

if phone_balance < 10 :
    phone_balance +=10
    wallet_balance -=10
elif phone_balance > 10:
    phone_balance -=10
    wallet_balance +=10
else:
    exit
# print(phone_balance, wallet_balance)

# season = 'summer'
# if season == 'spring':
#     print('plant the garden!')
# elif season == 'summer':
#     print('water the garden!')
# elif season == 'fall':
#     print('harvest the garden!')
# elif season == 'winter':
#     print('stay indoors!')
# else:
#     print('unrecognized season')

points = 174
if points>= 1 and  points <= 50:
    print('Congratulations! You won a wooden rabbit')
elif points>=51 and  points <= 150:
    print('Oh dear, no prize this time.')
elif points>=151 and  points <= 180:
    print('Congratulations! You won a wafer-thin mint')
elif points>=181 and  points <= 200:
    print('Congratulations! You won a penguin')
else:
    exit

