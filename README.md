# Assignment-3-Advanced-Python
## Name: Saloni Dixon 
## Programming Language: Python
## Date: 10/05/23
### Description 
For this assignment, a python script was created in order to implement additional practice with the fundamental commands in base Python, understand how to create custom Python functions, utilize numpy, pandas, and seaborn for data handling, create graphs of many kinds and understand the many parameters available to present a professional and publishable graph, and managing a Git repository.

### Required Files: 
tit_test: CSV testing data from the Titanic Survival Predication Dataset

tit_train: CSV training data from the Titanic Survival Predication Dataset

### Required Packages: 
matplotlib.pyplot: a collection of functions that make matplotlib function like MATLAB

seaborn: a Python data visualization library based on matplotlib

### Execution: 
1. Imported necessary files and packages for implementation of assignment
   
2. Wrote a function to take an integer, X, as an input and return a numpy array that contains the first X numbers of the first X numbers of the Fibonacci numbers.
```
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
```
3. Call function to generate the first 20 Fibonacci numbers and then print the result of the numpy array in order to create a numpy array with the first 20 values of the Fibonacci #'s
```
# call function to generate the first 20 fibonacci numbers
X = 20
fibonacci_array = fibonacci_numpy_array(X)

# print the result of numpy array
print("Result of First 20 Fibonacci Numbers: ", fibonacci_array)
```
4. Used numpy array from above to recreate arrays from part 3 (assignment #2) to create one array that is the quotient of consecutive Fibonacci numbers, and the difference of the quotient of consecutive Fibonacci numbers.
```
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
```
5. Plotted ALL 3 of the above-mentioned serioes on the same graph.
```
indcs = np.arange(X)   # create an array for the indices of the sequences

# plot ALL 3 series on the same graph
plt.plot(indcs, fibonacci_array, label = 'Fibonacci Sequence', marker = 'o')
plt.plot(indcs, quotient_list, label = "Quotient Array", marker = 'x')
plt.plot(indcs, difference_list, label = 'Difference Array', marker = 's')

plt.xlabel('Index')   # set the label on x-axis
plt.ylabel('Value')   # set the label on y-axis
plt.title('Series of Fibonacci Sequence, Quotient, and Difference Array')   # set the title
plt.legend()          # add a legend

# show the plot
plt.show()
```
6. Based upon my observances, it seems that the quotient and difference array remain the same as the index increases. However, the Fibonacci Sequence gradually increases in value as the index increases after the index of 9. 
7. Downloaded the testing and training data on the 'Titanic Disaster' or Titanic Survival Predication Dataset from the Kaggle Study.
8. Opened both files on pandas dataframes and concatenated them together.
```
# load the training and testing datasets into pandas df
# absolute path from my file directory was used (due to issues with relative path)
tit_train_df = pd.read_csv("C:/Users/justi/OneDrive/Desktop/INFO-B 473/Assignment #3/Titanic Dataset/tit_train.csv")
tit_test_df = pd.read_csv("C:/Users/justi/OneDrive/Desktop/INFO-B 473/Assignment #3/Titanic Dataset/tit_test.csv")

# concatenating training & testing df
tit_concat_df = pd.concat([tit_train_df, tit_test_df], ignore_index = True)
```
9. Created a summary of the pandas dataframe.
```
print(tit_concat_df.info())
print(tit_concat_df.describe())
```
10. Created 2 Histograms: one showing the distribution of age of people on the Titanic and another one showing the distribution of age of people on the Titanic segregated by survivalship.
```
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
```
11. Created 2 Bar Charts: one showing the percentages of who survived on the Titanic and another one showing the percentages of who survived on the Titanic segregated by sex.
```
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
```
13. Created 2 Boxplots: one showing the distribution of who survived on the Titanic vs. their Passenger Class and then another one showing the distribution of who survived on the Titanic vs. their Passenger Class segregated by sex.
```
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
```
15. Based on my findings from the Titanic Dataset, it demonstrates the following for overall survivalship: 1.) Between the ages of 25 to 30 years old, this was the most likely age to not survive on the Titanic, 2.) Based on the graph of "Distribution of Ages on the Titanic", the age of 21 was the most common on the ship with the frequency of 160 people in comparison to 40 years old with the frequency of 50, and 3.) The survivalship for women was significantly higher than men by the percentage difference of 53%. Something I found to be very surprising from the various visualizations provided involved the "Distribution of Ages on the Titanic (Segregated by Survival" graph and how among those that did survive, the frequency was lower than those that did NOT survive and the age was higher (ex: 35 years old had a frequency of 32). 
