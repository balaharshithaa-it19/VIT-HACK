amount=5000
transaction ='y'

printf("1. Check Balance\n")
printf("2. Withdraw Cash\n")
printf("3. Deposit Cash\n")
printf("4. Quit\n")
printf("Enter your choice: ")
choice = int(input())
if (choice==1):
    {
        print("\n YOUR BALANCE IN Rs : "+ amount)
    }
elif (choice==2):
    {
        print("\n ENTER THE AMOUNT TO WITHDRAW: ")
        withdraw = int(input())
        {
            if (withdraw % 100 != 0)
            {
                print("\n PLEASE ENTER THE AMOUNT IN MULTIPLES OF 100")
            }
            elif (withdraw >(amount - 500))
            {
                print("\n INSUFFICENT BALANCE")
            }
            else
            {
                amount = amount - withdraw;
                print("\n YOUR CURRENT BALANCE IS "+ amount)
            }
        }
    }
elif (choice==3):
    {
        print("\n ENTER THE AMOUNT TO DEPOSIT")
        deposit = int(input())
        amount = amount + deposit
        print("YOUR BALANCE IS "+ amount)
    }
elif (choice==4):
    {
        print("\n THANK U ")
        
else:
    {
        print("\n INVALID CHOICE")
    }
{
print("\n\n\n DO U WISH TO HAVE ANOTHER TRANSCATION?(y/n): \n")
transaction = int(input())
if (transaction == 'n'|| transaction == 'N')
k = 1
} while (!k)
print("\n\n THANKS FOR USING....QUITING")

