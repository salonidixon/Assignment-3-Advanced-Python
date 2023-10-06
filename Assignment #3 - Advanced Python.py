# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:31:58 2023

@author: saloni dixon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# *** QUESTION #1 ***
# *** a.) write a function that takes an integer X as an input and returns a 
# numpy array that contains the first X numbers of the fibonacci numbers

def fibonacci_numpy_array(X):
    if X <= 0:
        return np.array([])    # return an empty array for an invalid input
    
    fibonacci_numbers = np.zeros(X, dtype = int)
    fibonacci_numbers[:2] = [0,1]
    
    for i in range(2, X):
        fibonacci_numbers[i] = fibonacci_numbers[i-1] + fibonacci_numbers[i-2]
    return fibonacci_numbers

# prompt the user to input an integer
try: 
    X = int(input("Enter the # of Fibonacci numbers to generate: "))
    result = fibonacci_numpy_array(X)
    print(result)
except ValueError: 
    print("Please enter a valid number. Try again.")


# *** b.) use fibonacci_numpy_array to create a numpy array with the first 20 
# values of the fibonacci #'s

# call function to generate the first 20 fibonacci numbers
X = 20
fibonacci_array = fibonacci_numpy_array(X)

# print the result of numpy array
print("Result of First 20 Fibonacci Numbers: ", fibonacci_array)

# *** c.) use numpy array above to recreate arrays from part 3 (assignment #2)
# one array that is the quotient of consecutive Fibonacci numbers, and 
# the difference of the quotient of consecutive Fibonacci numbers

# pulling functions previously created from past 3, assignment #2

# function to generate the quotient of consecutive Fibonacci numbers
def generate_quotient_list(fibonacci_sequence):
    quotient_list = np.ones(len(fibonacci_sequence))  # Initialize with ones

    for i in range(1, len(fibonacci_sequence)):
        # calculating the quotient of the current Fibonacci number and the
        # previous Fibonacci number
        quotient_list[i] = fibonacci_sequence[i] / fibonacci_sequence[i - 1]

    return quotient_list

# function to generate the difference of the quotient of consecutive Fibonacci numbers
def generate_difference_list(quotient_list):
    difference_list = np.zeros(len(quotient_list))  # Initialize with zeros

    for i in range(2, len(quotient_list)):
        # calculating the difference between the current quotient and the
        # previous element in the difference list
        difference_list[i] = quotient_list[i] - quotient_list[i - 1]

    return difference_list

# provided Fibonacci array from question 1, part b
X = 20
fibonacci_array = fibonacci_numpy_array(X)

# generating quotient & difference list
quotient_list = generate_quotient_list(fibonacci_array)
difference_list = generate_difference_list(quotient_list)

# print the quotient & difference arrays
print("Quotient Array:", quotient_list)
print("\nDifference Array:", difference_list)

# *** d.) plot all 3 of these series on the same graph (adjust parameters of 
# the plot to see all 3 series at once)

indcs = np.arange(X)   # create an array for the indices of the sequences

# plot ALL 3 series on the same graph
plt.plot(indcs, fibonacci_array, label = 'Fibonacci Sequence', marker = '8')
plt.plot(indcs, quotient_list, label = "Quotient Array", marker = '*')
plt.plot(indcs, difference_list, label = 'Difference Array', marker = '+')

plt.xlabel('Index')   # set the label on x-axis
plt.ylabel('Value')   # set the label on y-axis
plt.title('Series of Fibonacci Sequence, Quotient, and Difference Array')   # set the title
plt.legend()          # add a legend

# show the plot
plt.show()

# *** e.) based upon your observation, do any of these series appear to be 
# converging? If so what values do they appear to be converging to? feel free 
# to reference the values of the series when determining what value it appears 
# to be converging to.

# *** QUESTION #2 ***
# a.) download data on 'Titanic Disaster' from the Kaggle study (training & 
# testing data).
# downloaded through link & imported into file directory 

# b.) open both files & concatenate them together
import pandas as pd

# load the training and testing datasets into pandas df
tit_train_df = pd.read_csv("C:/Users/justi/OneDrive/Desktop/INFO-B 473/Assignment #3/Titanic Dataset/tit_train.csv")
tit_test_df = pd.read_csv("C:/Users/justi/OneDrive/Desktop/INFO-B 473/Assignment #3/Titanic Dataset/tit_test.csv")

# concatenating training & testing df
tit_concat_df = pd.concat([tit_train_df, tit_test_df], ignore_index = True)

# c.) create a summery of pandas df
print(tit_concat_df.info())
print(tit_concat_df.describe())

# d.) create 2 histograms: showing the distribution age of people on the titanic 
# and another histogram showing the distribution of age of people on the titanic 
# segregated by survivalship

# creating a histogram of the distribution of ages of ALL passengers
plt.hist(tit_concat_df['Age'].dropna(), bins=20, edgecolor='gold', color = 'yellow')
plt.title('Distribution of Ages on the Titanic')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# creating a histogram of the distribution of ages segregated by survival 
# 0 = did not survive
# 1 = survive
plt.hist(tit_concat_df[tit_concat_df['Survived'] == 0]['Age'].dropna(), bins=20, 
         alpha=0.5, linewidth=0.5, color='purple', label='Did Not Survive', edgecolor='mediumorchid')
plt.hist(tit_concat_df[tit_concat_df['Survived'] == 1]['Age'].dropna(), bins=20, 
         alpha=0.5, linewidth=0.5, color='green', label='Survived', edgecolor='darkolivegreen')
plt.title('Distribution of Ages on the Titanic (Segregated by Survival)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# e.) create 2 bar charts: showing the percentages of who survived on the titanic
# and another showing the percentages of who survived on the titanic segregated 
# by sex

# calculating survival percentage
survival_counts = tit_concat_df['Survived'].value_counts(normalize = True)*100

# creating a bar chart for surival percentages
colors = ['lightpink','peachpuff']
survival_counts.plot(kind='bar', color=colors, edgecolor = 'peru')
plt.title('Survival Percentage of the Titanic')
plt.xlabel('Outcome')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend()
plt.show()

# calculating survival percentage segregated by sex
sex_survival_counts = tit_concat_df.groupby('Sex')['Survived'].value_counts(normalize = True).unstack()*100

# creating a bar chart for survial percentages segregated by sex
sex_colors = ['aquamarine','lightsalmon']
sex_survival_counts.plot(kind='bar', color=sex_colors, edgecolor='peru')
plt.title('Survival Percentage on the Titanic (Segregated by Sex)')
plt.xlabel('Sex')
plt.ylabel('Percentage (%)')
plt.grid(axis='y', linestyle=':')
plt.legend(['Did Not Survive', 'Survived'], loc='upper left', title='Outcome')
plt.show()

# f.) create 2 boxplots: showing the distribution of who survived on the titanic 
# vs. their passenger class and another showing the distribution of who survived 
# on the titanic vs. their passenger class segregated by sex

# creating a boxplot for survival vs. passenger class
sns.set(style = "darkgrid")
sns.boxplot(x = 'Pclass', y='Age', data = tit_concat_df, palette = 'Set3')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.title('Distribution of Survival vs. Passenger Class')
plt.grid(axis='y', linestyle=':')
plt.show()

# creating a boxplot for survival vs. passenger class segregated by sex
sns.set(style = "darkgrid")
sns.boxplot(x='Pclass', y='Age', data = tit_concat_df, palette='Paired')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.title('Distribution of Survival vs. Passenger Class (Segregated by Sex)')
plt.legend(title='Sex', loc='upper right')
plt.grid(axis='y', linestyle=':')
plt.show()

# g.) write a few sentences explaining your analysis
# feel free to reference any visualizations

# *** QUESTION #3 *** 
# write a readme file meeting all the requirements in assignment #1



