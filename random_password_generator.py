#Random password generators
import random 
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  
CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                     'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                     'z''A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
                     'I', 'J', 'K', 'M', 'N', 'O', 'p', 'Q',
                     'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z']
  
SYMBOLS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>', 
           '*', '(', ')', '<']

print("Password generator!")
nr_letters=int(input("Letters in password?\n"))
nr_symbols=int(input("Symbols in password?\n"))
nr_digit=int(input("Numbers in password?\n"))
password=""

for character in range (0,nr_letters):
    password= password+random.choice(CHARACTERS)
    
for symbols in range(0,nr_symbols):
    password=password+random.choice(SYMBOLS)
    
for number in range(0,nr_digit):
    password=password+random.choice(DIGITS)
    
print(f"Your randomly generated password:{password}")

