# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ft_price_estimate.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gigregoi <gigregoi@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/09/16 19:15:14 by gigregoi          #+#    #+#              #
#    Updated: 2020/12/24 06:04:06 by gigregoi         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from ft_linear_regression import ft_linear_regression as MyLR
import numpy as np

if __name__ == "__main__":
    list_choix = [1,2]
    try :
        choix = int(input("Choisissez 1 pour la partie obligatoire, 2 pour la partie BONUS - Default value = 1\n"))
        if not(choix in list_choix):
            choix = 1
    except :
        choix = 1
    if (choix == 1):
        #  Predict car price from its mileage
        try:
            theta = np.load('./Res1Var/theta_output.npy')
        except:
            print("\33[31mCalculate regression coefficient before prediction\33[0m")
            theta = np.array([[0., 0.]])
        
        float_formatter = "{:.2e}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        print(f"The theta used for the prediction is " + '\33[92m' + f"{theta.flatten()}" + '\33[0m' )
        model = MyLR(theta)

        KM_str = input("Enter the amount of kilometers to estimate the price \n")
        try :
            KM = float(KM_str)
            price = str(round(np.ndarray.item(model.predict_(KM, theta))))
            print("Estimated price : {} $\n".format(price))
        except :
            print("\33[31mInput real number\33[0m")        

    if (choix == 2):
        print("Estimation of the price of the spacecraft")
        try:
            theta = np.load('./ResmultiVar/theta_output.npy')
        except:
            print("\33[31mCalculate regression coefficients before prediction\33[0m")
            theta = np.array([[0., 0., 0., 0.]])
        float_formatter = "{:.2e}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        print(f"The theta used for the prediction is " + '\33[92m' + f"{theta.flatten()}" + '\33[0m' )
        model = MyLR(theta)
        try : 
            Age = input("Enter the spacecraft Age to estimate the price, default = 5\n")
            Age = float(Age)
        except :
            Age = 5.0
        print("\33[100mAge of the spacecraft is:"+ f"{Age:.1f}\33[0m\n")
        try : 
            Thrust_power = input("Enter the spacecraft Thrust_power to estimate the price, default = 100 \n")
            Thrust_power = float(Thrust_power)
        except :
            Thrust_power = 100.0
        print("\33[100mThrust_power of the spacecraft is:"+ f"{Thrust_power:.1f}\33[0m")
        try : 
            Terameters = input("Enter the amount of Terameters to estimate the price, default = 50 \n")
            Terameters = float(Terameters)
        except :
            Terameters = 50.0
        print("\33[100mMileAge of the spacecraft is:"+ f"{Terameters:.1f}\33[0m\n")
        x0 = [Age, Thrust_power, Terameters]
        print("parameters stored : "+f"{x0}\n")
    
        try :
            price = str(round(np.ndarray.item(model.predict_(x0, theta))))
            print("\33[1m\33[94mEstimated price :" + f"{price} $\33[0m")
        except :
            print("\33[31mInput real number\33[0m") 
