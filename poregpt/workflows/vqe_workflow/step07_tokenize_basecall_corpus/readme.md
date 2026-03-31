validation_cyclone.csv 是王雪根据cyclone的模型进行打标的结果i


out_summary.processed.tsv 这个文件里的格式如下
filename        read_id chunk_start     chunk_size      alignment_identity
validation_00006.fast5  250F701901011_23_2561_1360_397804127_558818     3000    6000    1.0
validation_00006.fast5  250F701901011_23_2561_1360_397804127_558818     6000    6000    1.0
validation_00006.fast5  250F701901011_23_2561_1360_397804127_558818     9000    6000    1.0
validation_00006.fast5  250F701901011_23_2561_1360_397804127_558818     12000   6000    1.0
。references.npy里包含相同行数的np.array. 格式如下File: references.npy
Data Type: uint8
Shape: (537111, 762)
Number of dimensions: 2
Total number of elements: 409278582
Size in bytes: 409278582
请同时打开这两个文件，将npy里对应的数组读出来，把里面的0去掉。保存成12345这样的字符串， 然后增加一项 bases 用来表达这些数字字符串。 最后保存成csv: 表头是fast5, read_id, chunk_start, chunk_size, alignment_identity, bases


