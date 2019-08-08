


def inverse_matrix(matrix):    
    n = len(matrix)
    inverse = [[1 if i==j else 0 for i in range(n)] for j in range(n)]
    for i in range(0,n):
        m = matrix[i][i]*1.0
        #print("multiplier = ", m)
        for j in range(0,n):
            matrix[i][j] /= m
            inverse[i][j] /= m
        for j in range(i+1,n):
            m = matrix[j][i]
            for k in range(0,n):
                matrix[j][k] = (matrix[j][k] / m) - matrix[i][k]
                inverse[j][k] = (inverse[j][k] / m) - inverse[i][k]
        
    for i in range(n-1,-1,-1):
        
        for j in range(i-1,-1,-1):
            m = matrix[j][i]*1.0
            for k in range(0,n,1):                
                matrix[j][k] -= matrix[i][k] * m
                inverse[j][k] -= inverse[i][k] * m
    return inverse

def matrix_multiply(A,B):
    r1,c1 = len(A),len(A[0])    
    r2,c2 = len(B), len(B[0])
    if(c1 != r2):
        print("Array shapes are not compatible \n")
        return
    C = [[0 for i in range(c2)] for j in range(r1)]
    for i in range(0,r1):
        for j in range(0,c2):
            for k in range(0,r2):
                C[i][j] += 1.0*A[i][k]*B[k][j]
                
    return C

def transpose_matrix(A):
    if type(A[0]) is not list:
        return A
    r,c = len(A),len(A[0])
    B = [[0.0 for i in range(r)] for j in range(c)]
    for i in range(0,c):
        for j in range(0,r):
            B[i][j] = A[j][i]
    return B

"""Expects input a list of lists"""
def increase_dimension_2(x_data):
    n = len(x_data[0])
    X_new = []
    for x in x_data:
        #x_new = [0 for p in range((2+n-1)*(n)/2)]
        x_new = dict()
        for i in range(n):
            for j in range(n):
                index = str(min(i,j))+" "+str(max(i,j))
                if index not in x_new:
                    x_new[index] = 0
                value = x[i]*x[j]
                x_new[index] += value
        x_ = []
        for key,value in x_new.items():
            x_.append(value)
        X_new.append(x_)

    return X_new                
        
                
        


"""Data will always be of the form |x1,x2,..,xd|y
   This method adds 1 as the first value"""
def read_data(file_name):
    x_loc , y_loc = [],-1
    x_data = []
    y_data = []
    with open(file_name,"r") as fp:
        count = 0
        for line in fp:
            if(count == 0):
                count += 1
                larray = line.split(",")
                loc = 0
                for i in larray:
                    if i[0] == 'x':
                        x_loc.append(loc)
                    if i[0] == 'y':
                        y_loc = loc
                    loc += 1
            else:
                larray = line.split(",")
                x_att = []
                x_att.append(1)
                for i in x_loc:
                    x_att.append(float(larray[i]))
                x_data.append(x_att)
                y_data.append([float(larray[y_loc])])
    return (x_data, y_data)



def create_regression_model(test_data_file_name):  
    x_data, y_data = read_data(test_data_file_name)
    print("-----------------")
    x_data_new = increase_dimension_2(x_data)
    x_transpose = transpose_matrix(x_data_new)
    x_plus = matrix_multiply(x_transpose, x_data_new)
    x_plus_inv = inverse_matrix(x_plus)
    x_plus2 = matrix_multiply(x_plus_inv, x_transpose)
    W = matrix_multiply(x_plus2,y_data)
    
    return W


def validate_model(validate_data_file, W):
    accuracy = 0.0
    x_validate_data, y_validate_data = read_data(validate_data_file)
    N = len(x_validate_data)
    x_validate_data_new = increase_dimension_2(x_validate_data)
    y_output = matrix_multiply(x_validate_data_new,W)
    for i in range(N):
        label = 0
        if y_output[i][0] > 0:
            label = +1
        else:
            label = -1
        if label == y_validate_data[i][0]:
            accuracy += 1
    print("accuracy = %f"%(accuracy*100.0/N))
    return accuracy
    
test_data_file = "./non_linear_separable_data_labelled01_train_data.csv"
validation_data_file = "./non_linear_separable_data_labelled01_validation_data.csv"

W = create_regression_model(test_data_file)
validate_model(validation_data_file, W)

