??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
: *
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
: *
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:@*
dtype0
{
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_45/kernel
t
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes
:	?@*
dtype0
r
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_45/bias
k
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes
:@*
dtype0
z
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
* 
shared_namedense_46/kernel
s
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes

:@
*
dtype0
r
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_46/bias
k
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_22/kernel/m
?
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_22/bias/m
{
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_23/kernel/m
?
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_45/kernel/m
?
*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_45/bias/m
y
(Adam/dense_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_46/kernel/m
?
*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m*
_output_shapes

:@
*
dtype0
?
Adam/dense_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_46/bias/m
y
(Adam/dense_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_22/kernel/v
?
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_22/bias/v
{
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_23/kernel/v
?
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_45/kernel/v
?
*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_45/bias/v
y
(Adam/dense_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_46/kernel/v
?
*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v*
_output_shapes

:@
*
dtype0
?
Adam/dense_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_46/bias/v
y
(Adam/dense_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
8
0
1
2
3
&4
'5
,6
-7
 
8
0
1
2
3
&4
'5
,6
-7
?

7layers
	trainable_variables
8metrics
9layer_regularization_losses
:layer_metrics

regularization_losses
	variables
;non_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

<layers
=metrics
>layer_regularization_losses
?layer_metrics
trainable_variables
	variables
@non_trainable_variables
 
 
 
?
regularization_losses

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
trainable_variables
	variables
Enon_trainable_variables
\Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
trainable_variables
	variables
Jnon_trainable_variables
 
 
 
?
regularization_losses

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
 	variables
Onon_trainable_variables
 
 
 
?
"regularization_losses

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
#trainable_variables
$	variables
Tnon_trainable_variables
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_45/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
?
(regularization_losses

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
)trainable_variables
*	variables
Ynon_trainable_variables
[Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_46/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
.regularization_losses

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
/trainable_variables
0	variables
^non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
}
VARIABLE_VALUEAdam/conv2d_22/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_22_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_22_inputconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_45/kerneldense_45/biasdense_46/kerneldense_46/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_312610
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOp#dense_46/kernel/Read/ReadVariableOp!dense_46/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp(Adam/dense_45/bias/m/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp(Adam/dense_46/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp(Adam/dense_45/bias/v/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOp(Adam/dense_46/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_312977
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_45/kerneldense_45/biasdense_46/kerneldense_46/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/dense_45/kernel/mAdam/dense_45/bias/mAdam/dense_46/kernel/mAdam/dense_46/bias/mAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/dense_45/kernel/vAdam/dense_45/bias/vAdam/dense_46/kernel/vAdam/dense_46/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_313086??
?

?
.__inference_sequential_17_layer_call_fn_312381
conv2d_22_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_3123622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?
M
1__inference_max_pooling2d_22_layer_call_fn_312754

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312804

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?I
?
__inference__traced_save_312977
file_prefix/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop.
*savev2_dense_46_kernel_read_readvariableop,
(savev2_dense_46_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableop3
/savev2_adam_dense_45_bias_m_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop3
/savev2_adam_dense_46_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableop3
/savev2_adam_dense_45_bias_v_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop3
/savev2_adam_dense_46_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop*savev2_dense_46_kernel_read_readvariableop(savev2_dense_46_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop/savev2_adam_dense_45_bias_m_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop/savev2_adam_dense_46_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop/savev2_adam_dense_45_bias_v_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableop/savev2_adam_dense_46_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:	?@:@:@
:
: : : : : : : : : : : : @:@:	?@:@:@
:
: : : @:@:	?@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@:$  

_output_shapes

:@
: !

_output_shapes
:
:"

_output_shapes
: 
?
?
D__inference_dense_46_layer_call_and_return_conditional_losses_312355

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
.__inference_sequential_17_layer_call_fn_312652

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_3124872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312253

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312231

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_45_layer_call_fn_312824

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_3123382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_312307

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?"
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312581
conv2d_22_input*
conv2d_22_312557: 
conv2d_22_312559: *
conv2d_23_312563: @
conv2d_23_312565:@"
dense_45_312570:	?@
dense_45_312572:@!
dense_46_312575:@

dense_46_312577:

identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallconv2d_22_inputconv2d_22_312557conv2d_22_312559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_3122842#
!conv2d_22/StatefulPartitionedCall?
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122942"
 max_pooling2d_22/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_312563conv2d_23_312565*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_3123072#
!conv2d_23/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3123172"
 max_pooling2d_23/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_3123252
flatten_17/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_45_312570dense_45_312572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_3123382"
 dense_45/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_312575dense_46_312577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_3123552"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?

?
.__inference_sequential_17_layer_call_fn_312631

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_3123622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_22_layer_call_fn_312749

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_312784

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_45_layer_call_and_return_conditional_losses_312338

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312294

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_312325

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
.__inference_sequential_17_layer_call_fn_312527
conv2d_22_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_3124872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?
G
+__inference_flatten_17_layer_call_fn_312809

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_3123252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_46_layer_call_and_return_conditional_losses_312855

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_45_layer_call_and_return_conditional_losses_312835

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_312284

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?"
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312487

inputs*
conv2d_22_312463: 
conv2d_22_312465: *
conv2d_23_312469: @
conv2d_23_312471:@"
dense_45_312476:	?@
dense_45_312478:@!
dense_46_312481:@

dense_46_312483:

identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_22_312463conv2d_22_312465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_3122842#
!conv2d_22/StatefulPartitionedCall?
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122942"
 max_pooling2d_22/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_312469conv2d_23_312471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_3123072#
!conv2d_23/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3123172"
 max_pooling2d_23/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_3123252
flatten_17/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_45_312476dense_45_312478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_3123382"
 dense_45/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_312481dense_46_312483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_3123552"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_313086
file_prefix;
!assignvariableop_conv2d_22_kernel: /
!assignvariableop_1_conv2d_22_bias: =
#assignvariableop_2_conv2d_23_kernel: @/
!assignvariableop_3_conv2d_23_bias:@5
"assignvariableop_4_dense_45_kernel:	?@.
 assignvariableop_5_dense_45_bias:@4
"assignvariableop_6_dense_46_kernel:@
.
 assignvariableop_7_dense_46_bias:
&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: E
+assignvariableop_17_adam_conv2d_22_kernel_m: 7
)assignvariableop_18_adam_conv2d_22_bias_m: E
+assignvariableop_19_adam_conv2d_23_kernel_m: @7
)assignvariableop_20_adam_conv2d_23_bias_m:@=
*assignvariableop_21_adam_dense_45_kernel_m:	?@6
(assignvariableop_22_adam_dense_45_bias_m:@<
*assignvariableop_23_adam_dense_46_kernel_m:@
6
(assignvariableop_24_adam_dense_46_bias_m:
E
+assignvariableop_25_adam_conv2d_22_kernel_v: 7
)assignvariableop_26_adam_conv2d_22_bias_v: E
+assignvariableop_27_adam_conv2d_23_kernel_v: @7
)assignvariableop_28_adam_conv2d_23_bias_v:@=
*assignvariableop_29_adam_dense_45_kernel_v:	?@6
(assignvariableop_30_adam_dense_45_bias_v:@<
*assignvariableop_31_adam_dense_46_kernel_v:@
6
(assignvariableop_32_adam_dense_46_bias_v:

identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_45_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_45_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_46_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_46_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_22_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_22_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_23_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_23_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_45_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_45_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_46_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_46_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_22_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_22_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_23_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_23_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_45_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_45_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_46_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_46_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?=
?
!__inference__wrapped_model_312222
conv2d_22_inputP
6sequential_17_conv2d_22_conv2d_readvariableop_resource: E
7sequential_17_conv2d_22_biasadd_readvariableop_resource: P
6sequential_17_conv2d_23_conv2d_readvariableop_resource: @E
7sequential_17_conv2d_23_biasadd_readvariableop_resource:@H
5sequential_17_dense_45_matmul_readvariableop_resource:	?@D
6sequential_17_dense_45_biasadd_readvariableop_resource:@G
5sequential_17_dense_46_matmul_readvariableop_resource:@
D
6sequential_17_dense_46_biasadd_readvariableop_resource:

identity??.sequential_17/conv2d_22/BiasAdd/ReadVariableOp?-sequential_17/conv2d_22/Conv2D/ReadVariableOp?.sequential_17/conv2d_23/BiasAdd/ReadVariableOp?-sequential_17/conv2d_23/Conv2D/ReadVariableOp?-sequential_17/dense_45/BiasAdd/ReadVariableOp?,sequential_17/dense_45/MatMul/ReadVariableOp?-sequential_17/dense_46/BiasAdd/ReadVariableOp?,sequential_17/dense_46/MatMul/ReadVariableOp?
-sequential_17/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_22/Conv2D/ReadVariableOp?
sequential_17/conv2d_22/Conv2DConv2Dconv2d_22_input5sequential_17/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
sequential_17/conv2d_22/Conv2D?
.sequential_17/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_22/BiasAdd/ReadVariableOp?
sequential_17/conv2d_22/BiasAddBiasAdd'sequential_17/conv2d_22/Conv2D:output:06sequential_17/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
sequential_17/conv2d_22/BiasAdd?
sequential_17/conv2d_22/ReluRelu(sequential_17/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_17/conv2d_22/Relu?
&sequential_17/max_pooling2d_22/MaxPoolMaxPool*sequential_17/conv2d_22/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_22/MaxPool?
-sequential_17/conv2d_23/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_17/conv2d_23/Conv2D/ReadVariableOp?
sequential_17/conv2d_23/Conv2DConv2D/sequential_17/max_pooling2d_22/MaxPool:output:05sequential_17/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2 
sequential_17/conv2d_23/Conv2D?
.sequential_17/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_23/BiasAdd/ReadVariableOp?
sequential_17/conv2d_23/BiasAddBiasAdd'sequential_17/conv2d_23/Conv2D:output:06sequential_17/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2!
sequential_17/conv2d_23/BiasAdd?
sequential_17/conv2d_23/ReluRelu(sequential_17/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_17/conv2d_23/Relu?
&sequential_17/max_pooling2d_23/MaxPoolMaxPool*sequential_17/conv2d_23/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_23/MaxPool?
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2 
sequential_17/flatten_17/Const?
 sequential_17/flatten_17/ReshapeReshape/sequential_17/max_pooling2d_23/MaxPool:output:0'sequential_17/flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_17/flatten_17/Reshape?
,sequential_17/dense_45/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_45_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02.
,sequential_17/dense_45/MatMul/ReadVariableOp?
sequential_17/dense_45/MatMulMatMul)sequential_17/flatten_17/Reshape:output:04sequential_17/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_17/dense_45/MatMul?
-sequential_17/dense_45/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_17/dense_45/BiasAdd/ReadVariableOp?
sequential_17/dense_45/BiasAddBiasAdd'sequential_17/dense_45/MatMul:product:05sequential_17/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_17/dense_45/BiasAdd?
sequential_17/dense_45/ReluRelu'sequential_17/dense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_17/dense_45/Relu?
,sequential_17/dense_46/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_46_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02.
,sequential_17/dense_46/MatMul/ReadVariableOp?
sequential_17/dense_46/MatMulMatMul)sequential_17/dense_45/Relu:activations:04sequential_17/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_17/dense_46/MatMul?
-sequential_17/dense_46/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_46_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_17/dense_46/BiasAdd/ReadVariableOp?
sequential_17/dense_46/BiasAddBiasAdd'sequential_17/dense_46/MatMul:product:05sequential_17/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_46/BiasAdd?
sequential_17/dense_46/SoftmaxSoftmax'sequential_17/dense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_46/Softmax?
IdentityIdentity(sequential_17/dense_46/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp/^sequential_17/conv2d_22/BiasAdd/ReadVariableOp.^sequential_17/conv2d_22/Conv2D/ReadVariableOp/^sequential_17/conv2d_23/BiasAdd/ReadVariableOp.^sequential_17/conv2d_23/Conv2D/ReadVariableOp.^sequential_17/dense_45/BiasAdd/ReadVariableOp-^sequential_17/dense_45/MatMul/ReadVariableOp.^sequential_17/dense_46/BiasAdd/ReadVariableOp-^sequential_17/dense_46/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2`
.sequential_17/conv2d_22/BiasAdd/ReadVariableOp.sequential_17/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_22/Conv2D/ReadVariableOp-sequential_17/conv2d_22/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_23/BiasAdd/ReadVariableOp.sequential_17/conv2d_23/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_23/Conv2D/ReadVariableOp-sequential_17/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_17/dense_45/BiasAdd/ReadVariableOp-sequential_17/dense_45/BiasAdd/ReadVariableOp2\
,sequential_17/dense_45/MatMul/ReadVariableOp,sequential_17/dense_45/MatMul/ReadVariableOp2^
-sequential_17/dense_46/BiasAdd/ReadVariableOp-sequential_17/dense_46/BiasAdd/ReadVariableOp2\
,sequential_17/dense_46/MatMul/ReadVariableOp,sequential_17/dense_46/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?"
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312554
conv2d_22_input*
conv2d_22_312530: 
conv2d_22_312532: *
conv2d_23_312536: @
conv2d_23_312538:@"
dense_45_312543:	?@
dense_45_312545:@!
dense_46_312548:@

dense_46_312550:

identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallconv2d_22_inputconv2d_22_312530conv2d_22_312532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_3122842#
!conv2d_22/StatefulPartitionedCall?
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122942"
 max_pooling2d_22/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_312536conv2d_23_312538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_3123072#
!conv2d_23/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3123172"
 max_pooling2d_23/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_3123252
flatten_17/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_45_312543dense_45_312545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_3123382"
 dense_45/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_312548dense_46_312550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_3123552"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?
?
*__inference_conv2d_23_layer_call_fn_312773

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_3123072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_23_layer_call_fn_312794

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3123172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_312815

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312759

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312799

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312317

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?/
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312688

inputsB
(conv2d_22_conv2d_readvariableop_resource: 7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource: @7
)conv2d_23_biasadd_readvariableop_resource:@:
'dense_45_matmul_readvariableop_resource:	?@6
(dense_45_biasadd_readvariableop_resource:@9
'dense_46_matmul_readvariableop_resource:@
6
(dense_46_biasadd_readvariableop_resource:

identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2Dinputs'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_22/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_23/Relu?
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_17/Const?
flatten_17/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_17/Reshape?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMulflatten_17/Reshape:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_45/BiasAdds
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_45/Relu?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_46/BiasAdd|
dense_46/SoftmaxSoftmaxdense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_46/Softmaxu
IdentityIdentitydense_46/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_312610
conv2d_22_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@

	unknown_6:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3122222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_22_input
?"
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312362

inputs*
conv2d_22_312285: 
conv2d_22_312287: *
conv2d_23_312308: @
conv2d_23_312310:@"
dense_45_312339:	?@
dense_45_312341:@!
dense_46_312356:@

dense_46_312358:

identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_22_312285conv2d_22_312287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_3122842#
!conv2d_22/StatefulPartitionedCall?
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_3122942"
 max_pooling2d_22/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_312308conv2d_23_312310*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_3123072#
!conv2d_23/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3123172"
 max_pooling2d_23/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_3123252
flatten_17/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_45_312339dense_45_312341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_3123382"
 dense_45/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_312356dense_46_312358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_3123552"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312764

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_22_layer_call_fn_312733

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_3122842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
)__inference_dense_46_layer_call_fn_312844

inputs
unknown:@

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_3123552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_23_layer_call_fn_312789

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_3122532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_312744

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?/
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312724

inputsB
(conv2d_22_conv2d_readvariableop_resource: 7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource: @7
)conv2d_23_biasadd_readvariableop_resource:@:
'dense_45_matmul_readvariableop_resource:	?@6
(dense_45_biasadd_readvariableop_resource:@9
'dense_46_matmul_readvariableop_resource:@
6
(dense_46_biasadd_readvariableop_resource:

identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2Dinputs'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_22/Relu?
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_23/Relu?
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPoolu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_17/Const?
flatten_17/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_17/Reshape?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMulflatten_17/Reshape:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_45/BiasAdds
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_45/Relu?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_46/BiasAdd|
dense_46/SoftmaxSoftmaxdense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_46/Softmaxu
IdentityIdentitydense_46/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????  : : : : : : : : 2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_22_input@
!serving_default_conv2d_22_input:0?????????  <
dense_460
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:Ӎ
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
?

7layers
	trainable_variables
8metrics
9layer_regularization_losses
:layer_metrics

regularization_losses
	variables
;non_trainable_variables
{__call__
z_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_22/kernel
: 2conv2d_22/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

<layers
=metrics
>layer_regularization_losses
?layer_metrics
trainable_variables
	variables
@non_trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
trainable_variables
	variables
Enon_trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_23/kernel
:@2conv2d_23/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
trainable_variables
	variables
Jnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
 	variables
Onon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
#trainable_variables
$	variables
Tnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_45/kernel
:@2dense_45/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
(regularization_losses

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
)trainable_variables
*	variables
Ynon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@
2dense_46/kernel
:
2dense_46/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
.regularization_losses

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
/trainable_variables
0	variables
^non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metric
^
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
/:- 2Adam/conv2d_22/kernel/m
!: 2Adam/conv2d_22/bias/m
/:- @2Adam/conv2d_23/kernel/m
!:@2Adam/conv2d_23/bias/m
':%	?@2Adam/dense_45/kernel/m
 :@2Adam/dense_45/bias/m
&:$@
2Adam/dense_46/kernel/m
 :
2Adam/dense_46/bias/m
/:- 2Adam/conv2d_22/kernel/v
!: 2Adam/conv2d_22/bias/v
/:- @2Adam/conv2d_23/kernel/v
!:@2Adam/conv2d_23/bias/v
':%	?@2Adam/dense_45/kernel/v
 :@2Adam/dense_45/bias/v
&:$@
2Adam/dense_46/kernel/v
 :
2Adam/dense_46/bias/v
?B?
!__inference__wrapped_model_312222conv2d_22_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_sequential_17_layer_call_fn_312381
.__inference_sequential_17_layer_call_fn_312631
.__inference_sequential_17_layer_call_fn_312652
.__inference_sequential_17_layer_call_fn_312527?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312688
I__inference_sequential_17_layer_call_and_return_conditional_losses_312724
I__inference_sequential_17_layer_call_and_return_conditional_losses_312554
I__inference_sequential_17_layer_call_and_return_conditional_losses_312581?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_22_layer_call_fn_312733?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_312744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_22_layer_call_fn_312749
1__inference_max_pooling2d_22_layer_call_fn_312754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312759
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_23_layer_call_fn_312773?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_312784?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_23_layer_call_fn_312789
1__inference_max_pooling2d_23_layer_call_fn_312794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312799
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312804?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_17_layer_call_fn_312809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_17_layer_call_and_return_conditional_losses_312815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_45_layer_call_fn_312824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_45_layer_call_and_return_conditional_losses_312835?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_46_layer_call_fn_312844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_46_layer_call_and_return_conditional_losses_312855?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_312610conv2d_22_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_312222?&',-@?=
6?3
1?.
conv2d_22_input?????????  
? "3?0
.
dense_46"?
dense_46?????????
?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_312744l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_22_layer_call_fn_312733_7?4
-?*
(?%
inputs?????????  
? " ?????????? ?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_312784l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_23_layer_call_fn_312773_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_dense_45_layer_call_and_return_conditional_losses_312835]&'0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_45_layer_call_fn_312824P&'0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense_46_layer_call_and_return_conditional_losses_312855\,-/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????

? |
)__inference_dense_46_layer_call_fn_312844O,-/?,
%?"
 ?
inputs?????????@
? "??????????
?
F__inference_flatten_17_layer_call_and_return_conditional_losses_312815a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
+__inference_flatten_17_layer_call_fn_312809T7?4
-?*
(?%
inputs?????????@
? "????????????
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312759?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
L__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_312764h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
1__inference_max_pooling2d_22_layer_call_fn_312749?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
1__inference_max_pooling2d_22_layer_call_fn_312754[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312799?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_312804h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
1__inference_max_pooling2d_23_layer_call_fn_312789?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
1__inference_max_pooling2d_23_layer_call_fn_312794[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312554{&',-H?E
>?;
1?.
conv2d_22_input?????????  
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312581{&',-H?E
>?;
1?.
conv2d_22_input?????????  
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312688r&',-??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_312724r&',-??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
.__inference_sequential_17_layer_call_fn_312381n&',-H?E
>?;
1?.
conv2d_22_input?????????  
p 

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_312527n&',-H?E
>?;
1?.
conv2d_22_input?????????  
p

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_312631e&',-??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_312652e&',-??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
$__inference_signature_wrapper_312610?&',-S?P
? 
I?F
D
conv2d_22_input1?.
conv2d_22_input?????????  "3?0
.
dense_46"?
dense_46?????????
