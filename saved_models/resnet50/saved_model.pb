´
£
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8
|
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_61/kernel
u
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel* 
_output_shapes
:
*
dtype0
s
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_61/bias
l
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes	
:*
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
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0

Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_61/kernel/m

*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/m
z
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_61/kernel/v

*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/v
z
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ò
valueÈBÅ B¾
¥
layer-0
layer_with_weights-0
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
R
	trainable_variables

regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_ratem>m?v@vA

0
1
 

0
1
­
layer_regularization_losses

layers
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
	variables
 
 
 
 
­
layer_regularization_losses

layers
non_trainable_variables
	trainable_variables

regularization_losses
 layer_metrics
!metrics
	variables
[Y
VARIABLE_VALUEdense_61/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_61/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
"layer_regularization_losses

#layers
$non_trainable_variables
trainable_variables
regularization_losses
%layer_metrics
&metrics
	variables
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
 

0
1
 
 

'0
(1
)2
*3
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
	+total
	,count
-	variables
.	keras_api
D
	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api
W
4
thresholds
5true_positives
6false_positives
7	variables
8	keras_api
W
9
thresholds
:true_positives
;false_negatives
<	variables
=	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

-	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

2	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

<	variables
~|
VARIABLE_VALUEAdam/dense_61/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_38Placeholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿ
Ü
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_38dense_61/kerneldense_61/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1538381
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1538525
Ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_61/kerneldense_61/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/dense_61/kernel/vAdam/dense_61/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1538592Ã²
ú.
ß
 __inference__traced_save_1538525
file_prefix.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¨	
value	B	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesó
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*z
_input_shapesi
g: :
:: : : : : : : : : :::::
::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 

´
"__inference__wrapped_model_1538251
input_38I
5sequential_41_dense_61_matmul_readvariableop_resource:
E
6sequential_41_dense_61_biasadd_readvariableop_resource:	
identity¢-sequential_41/dense_61/BiasAdd/ReadVariableOp¢,sequential_41/dense_61/MatMul/ReadVariableOpÕ
@sequential_41/global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_41/global_average_pooling2d_26/Mean/reduction_indicesð
.sequential_41/global_average_pooling2d_26/MeanMeaninput_38Isequential_41/global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_41/global_average_pooling2d_26/MeanÔ
,sequential_41/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_41_dense_61_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_41/dense_61/MatMul/ReadVariableOpê
sequential_41/dense_61/MatMulMatMul7sequential_41/global_average_pooling2d_26/Mean:output:04sequential_41/dense_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_41/dense_61/MatMulÒ
-sequential_41/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_41_dense_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_41/dense_61/BiasAdd/ReadVariableOpÞ
sequential_41/dense_61/BiasAddBiasAdd'sequential_41/dense_61/MatMul:product:05sequential_41/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_41/dense_61/BiasAdd§
sequential_41/dense_61/SoftmaxSoftmax'sequential_41/dense_61/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_41/dense_61/SoftmaxÜ
IdentityIdentity(sequential_41/dense_61/Softmax:softmax:0.^sequential_41/dense_61/BiasAdd/ReadVariableOp-^sequential_41/dense_61/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2^
-sequential_41/dense_61/BiasAdd/ReadVariableOp-sequential_41/dense_61/BiasAdd/ReadVariableOp2\
,sequential_41/dense_61/MatMul/ReadVariableOp,sequential_41/dense_61/MatMul/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
è
t
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_1538258

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Y
=__inference_global_average_pooling2d_26_layer_call_fn_1538264

inputs
identityâ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_15382582
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


%__inference_signature_wrapper_1538381
input_38
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_15382512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
Á

/__inference_sequential_41_layer_call_fn_1538416

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_15382902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

/__inference_sequential_41_layer_call_fn_1538425

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_15383282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
Õ
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538364
input_38$
dense_61_1538358:

dense_61_1538360:	
identity¢ dense_61/StatefulPartitionedCall
+global_average_pooling2d_26/PartitionedCallPartitionedCallinput_38*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_15382582-
+global_average_pooling2d_26/PartitionedCallÉ
 dense_61/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_61_1538358dense_61_1538360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_15382832"
 dense_61/StatefulPartitionedCall¡
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0!^dense_61/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
Ç
¡
/__inference_sequential_41_layer_call_fn_1538344
input_38
unknown:

	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_15383282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
®R
ð

#__inference__traced_restore_1538592
file_prefix4
 assignvariableop_dense_61_kernel:
/
 assignvariableop_1_dense_61_bias:	&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: 0
"assignvariableop_11_true_positives:1
#assignvariableop_12_false_positives:2
$assignvariableop_13_true_positives_1:1
#assignvariableop_14_false_negatives:>
*assignvariableop_15_adam_dense_61_kernel_m:
7
(assignvariableop_16_adam_dense_61_bias_m:	>
*assignvariableop_17_adam_dense_61_kernel_v:
7
(assignvariableop_18_adam_dense_61_bias_v:	
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¨	
value	B	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_61_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_61_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_true_positivesIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_false_positivesIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¬
AssignVariableOp_13AssignVariableOp$assignvariableop_13_true_positives_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_negativesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_61_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_61_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17²
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_61_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_61_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19ó
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
ð
¢
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538407

inputs;
'dense_61_matmul_readvariableop_resource:
7
(dense_61_biasadd_readvariableop_resource:	
identity¢dense_61/BiasAdd/ReadVariableOp¢dense_61/MatMul/ReadVariableOp¹
2global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_26/Mean/reduction_indicesÄ
 global_average_pooling2d_26/MeanMeaninputs;global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 global_average_pooling2d_26/Meanª
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_61/MatMul/ReadVariableOp²
dense_61/MatMulMatMul)global_average_pooling2d_26/Mean:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/MatMul¨
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_61/BiasAdd/ReadVariableOp¦
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/BiasAdd}
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/Softmax²
IdentityIdentitydense_61/Softmax:softmax:0 ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
¡
/__inference_sequential_41_layer_call_fn_1538297
input_38
unknown:

	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_38unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_15382902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
§

*__inference_dense_61_layer_call_fn_1538445

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_15382832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
Õ
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538354
input_38$
dense_61_1538348:

dense_61_1538350:	
identity¢ dense_61/StatefulPartitionedCall
+global_average_pooling2d_26/PartitionedCallPartitionedCallinput_38*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_15382582-
+global_average_pooling2d_26/PartitionedCallÉ
 dense_61/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_61_1538348dense_61_1538350*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_15382832"
 dense_61/StatefulPartitionedCall¡
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0!^dense_61/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_38
Á

ù
E__inference_dense_61_layer_call_and_return_conditional_losses_1538283

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
Ó
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538290

inputs$
dense_61_1538284:

dense_61_1538286:	
identity¢ dense_61/StatefulPartitionedCall
+global_average_pooling2d_26/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_15382582-
+global_average_pooling2d_26/PartitionedCallÉ
 dense_61/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_61_1538284dense_61_1538286*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_15382832"
 dense_61/StatefulPartitionedCall¡
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0!^dense_61/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

ù
E__inference_dense_61_layer_call_and_return_conditional_losses_1538436

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
¢
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538394

inputs;
'dense_61_matmul_readvariableop_resource:
7
(dense_61_biasadd_readvariableop_resource:	
identity¢dense_61/BiasAdd/ReadVariableOp¢dense_61/MatMul/ReadVariableOp¹
2global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_26/Mean/reduction_indicesÄ
 global_average_pooling2d_26/MeanMeaninputs;global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 global_average_pooling2d_26/Meanª
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_61/MatMul/ReadVariableOp²
dense_61/MatMulMatMul)global_average_pooling2d_26/Mean:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/MatMul¨
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_61/BiasAdd/ReadVariableOp¦
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/BiasAdd}
dense_61/SoftmaxSoftmaxdense_61/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_61/Softmax²
IdentityIdentitydense_61/Softmax:softmax:0 ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
Ó
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538328

inputs$
dense_61_1538322:

dense_61_1538324:	
identity¢ dense_61/StatefulPartitionedCall
+global_average_pooling2d_26/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_15382582-
+global_average_pooling2d_26/PartitionedCallÉ
 dense_61/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_61_1538322dense_61_1538324*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_15382832"
 dense_61/StatefulPartitionedCall¡
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0!^dense_61/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
F
input_38:
serving_default_input_38:0ÿÿÿÿÿÿÿÿÿ=
dense_611
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:q
¡
layer-0
layer_with_weights-0
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*B&call_and_return_all_conditional_losses
C_default_save_signature
D__call__"¢
_tf_keras_sequential{"name": "sequential_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 2048]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_38"}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 133, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 2048]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 1, 2048]}, "float32", "input_38"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 2048]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_38"}, "shared_object_id": 0}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 133, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 7}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 8}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 9}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ç
	trainable_variables

regularization_losses
	variables
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"¸
_tf_keras_layer{"name": "global_average_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
×

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"²
_tf_keras_layer{"name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 133, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
w
iter

beta_1

beta_2
	decay
learning_ratem>m?v@vA"
	optimizer
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ê
layer_regularization_losses

layers
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
	variables
D__call__
C_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
layer_regularization_losses

layers
non_trainable_variables
	trainable_variables

regularization_losses
 layer_metrics
!metrics
	variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_61/kernel
:2dense_61/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
"layer_regularization_losses

#layers
$non_trainable_variables
trainable_variables
regularization_losses
%layer_metrics
&metrics
	variables
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
'0
(1
)2
*3"
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
Ô
	+total
	,count
-	variables
.	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 11}

	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api"Ï
_tf_keras_metric´{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 7}
º
4
thresholds
5true_positives
6false_positives
7	variables
8	keras_api"à
_tf_keras_metricÅ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 8}
±
9
thresholds
:true_positives
;false_negatives
<	variables
=	keras_api"×
_tf_keras_metric¼{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 9}
:  (2total
:  (2count
.
+0
,1"
trackable_list_wrapper
-
-	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
-
2	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
:0
;1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
(:&
2Adam/dense_61/kernel/m
!:2Adam/dense_61/bias/m
(:&
2Adam/dense_61/kernel/v
!:2Adam/dense_61/bias/v
ö2ó
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538394
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538407
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538354
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538364À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
"__inference__wrapped_model_1538251À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_38ÿÿÿÿÿÿÿÿÿ
2
/__inference_sequential_41_layer_call_fn_1538297
/__inference_sequential_41_layer_call_fn_1538416
/__inference_sequential_41_layer_call_fn_1538425
/__inference_sequential_41_layer_call_fn_1538344À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_1538258à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
=__inference_global_average_pooling2d_26_layer_call_fn_1538264à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_dense_61_layer_call_and_return_conditional_losses_1538436¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_61_layer_call_fn_1538445¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
%__inference_signature_wrapper_1538381input_38"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"__inference__wrapped_model_1538251v:¢7
0¢-
+(
input_38ÿÿÿÿÿÿÿÿÿ
ª "4ª1
/
dense_61# 
dense_61ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_61_layer_call_and_return_conditional_losses_1538436^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_61_layer_call_fn_1538445Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿá
X__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_1538258R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
=__inference_global_average_pooling2d_26_layer_call_fn_1538264wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538354pB¢?
8¢5
+(
input_38ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¾
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538364pB¢?
8¢5
+(
input_38ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538394n@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_41_layer_call_and_return_conditional_losses_1538407n@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_41_layer_call_fn_1538297cB¢?
8¢5
+(
input_38ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_41_layer_call_fn_1538344cB¢?
8¢5
+(
input_38ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_41_layer_call_fn_1538416a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_41_layer_call_fn_1538425a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬
%__inference_signature_wrapper_1538381F¢C
¢ 
<ª9
7
input_38+(
input_38ÿÿÿÿÿÿÿÿÿ"4ª1
/
dense_61# 
dense_61ÿÿÿÿÿÿÿÿÿ