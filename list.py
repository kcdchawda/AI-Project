#list

nums=[10,20,30,40,999,50]
print(f"List befor the function{nums}")

nums.append(100)
print(f"List after the function'nums.append(100)' {nums}")

nums.insert(0,999)
print(f"List after the function'nums.insert(0,999)' {nums}")

nums.pop()
print(f"List after the function'nums.pop() '{nums}")

nums.pop(0)
print(f"List after the function'nums.pop(0) '{nums}")

length=len(nums)
print(f"List after the function' length=len(nums)'{length}")

dval =nums.pop()
print ("Delete is vlaue is:",dval)
print(nums)

dval =nums.pop(2)
print ("Delete is vlaue is:",dval)
print(nums)


