---
title: "Sed"
date: 2023-01-25T17:33:44+08:00
draft: false
category: linux
tags: ["linux", "command"]
ShowToc: true
---

## 概述

sed是stream editor的简称，也就是流编辑器。它一次处理一行内容，处理时，把当前处理的行存储在临时缓冲区中，称为"pattern space"，接着用sed命令处理缓冲区中的内容，处理完成后，把缓冲区的内容送往屏幕。接着处理下一行，这样不断重复，直到文件末尾。文件内容并没有 改变，除非你使用重定向存储输出。

## 基础

```shell
-E  -- use extended regular expressions
-a  -- delay opening files listed with w function
-e  -- specify sed commands to run
-f  -- add contents of file to commands to run
-i  -- edit files in-place, running scripts separately for each file
-l  -- make output line buffered
-n  -- suppress automatic printing of pattern space
```

```shell
a,append        追加
i,insert        插入
d,delete        删除
s,substitution  替换
c,change        修改
y,transform     转换
```

### 替换

#### 替换标记

`s`（substitution）默认情况下只替换每行中出现的第一处。使用替换标记（substitution flag）指定替换位置。替换标记在替换命令字符串之后设置。
`s/pattern/replacement/flags`
有4种可用的替换标记：

1. 数字，表明新文本将替换第几处模式匹配的地方；
2. g，表明新文本将会替换所有匹配的文本；
3. p，表明替换过的行要打印出来；
4. w file，将替换的结果写到文件中。

#### 替换字符

sed允许选择其他字符来作为替换命令中的字符串分隔符：

```shell
sed 's!/bin/bash!/bin/csh!' /etc/passwd
```

### 使用地址

>默认情况下，sed命令会作用于文本数据的所有行。如果只想将命令作用于特定行或某些行，则必须用行寻址（line addressing）。

- 以数字形式表示行区间
- 用文本模式来过滤出行
格式 `[address]command`

#### 以数字形式表示行区间

```shell
sed '2s/dog/cat/' data1.txt
sed '2,3s/dog/cat/' data1.txt
sed '2,$s/dog/cat/' data1.txt
```

#### 使用文本模式过滤器

sed编辑器允许指定文本模式来过滤出命令要作用的行
`/pattern/command`

```shell
sed -n '/special/s/test/dog/p' data1.txt
```

#### 命令组合

```shell
 sed '2{
> s/fox/elephant/
> s/dog/cat/
> }' data1.txt
```

### 删除行

删除命令`d`删除匹配指定寻址模式的所有行。如果没有加入寻址模式，流中的所有文本行都会被删除。

``` shell
sed '1d' data1.txt
sed '/spe/d' data1.txt
```

### 插入和附加文本

1. 插入（insert）命令（i）会在指定行前增加一个新行；
2. 附加（append）命令（a）会在指定行后增加一个新行。

```shell
sed '[address]command\
new line'
$ echo "Test Line 2" | sed 'i\Test Line 1'
Test Line 1
Test Line 2
$ echo "Test Line 2" | sed 'a\Test Line 1'
Test Line 2
Test Line 1
#在命令行上使用sed时，会看到次提示符来提醒输入新的行数据。必须在该行完成sed编辑器命令。
$ echo "Test Line 2" | sed 'i\
> Test Line 1'
Test Line 1
Test Line 2
```

### 修改行

修改（change）命令(c)允许修改(替换)数据流中整行文本的内容.如果作用于地址区间，sed会用一行文本代替区间内的所有行，而非逐一替换。

```shell
sed '1,2c\
hello
' data1.txt
```

### 转换命令

转换（transform）命令（y）是唯一可以处理单个字符的sed编辑器命令。转换命令格式如下:
`[address]y/inchars/outchars/`

### 回顾打印

3个命令也能用来打印数据流中的信息：
`p`命令用来打印文本行；
等号（`=`）命令用来打印行号；
`l`（小写的L）命令用来列出行。

### 使用sed处理文件

#### 写入文件

w命令用来向文件写入行 `[address]w filename`

#### 从文件读取数据

读取（read）命令（`r`）将一个独立文件中的数据插入到数据流中
`[address]r filename`

- https://jixiaocan.github.io/posts/cmd-sed/
- https://markrepo.github.io/commands/2018/06/26/sed/
