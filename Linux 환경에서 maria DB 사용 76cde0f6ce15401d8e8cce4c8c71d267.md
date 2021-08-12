# Linux 환경에서 maria DB 사용

![121](https://user-images.githubusercontent.com/54494622/129167283-911b7bd0-190d-4663-8b44-5ce91dfae584.png)
(Linux%20%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A6%E1%84%89%E1%85%A5%20maria%20DB%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%2076cde0f6ce15401d8e8cce4c8c71d267/121.png)

                                   to_char(sal,'999,999')                                          format

                                          테이블 생성 시 데이터 타입의 차이가 있습니다.

                                  csv 파일을 로드하려면                               load 명령어로 쉽게 입력

                                  sqldeveloper 를 사용

### ▩ 리눅스에 마리아 디비 설치 방법

[https://bamdule.tistory.com/59](https://bamdule.tistory.com/59)

```sql
mysql>
select ename,deptno,sal,if(deptno=10,300,else if(deptno=20,500,0)) bonus
from emp;

mysql>
SELECT ename,deptno,sal,
		if(deptno=10,300,if(deptno=20,500,0)) AS bonus
FROM emp;
```

```sql
mysql>
SELECT ename,deptno,sal,
		case
			when deptno=10 THEN 5000
			when deptno=20 THEN 6000
			ELSE 0
		END AS bonus
FROM emp;
```

```sql
mysql>
SELECT SUM(if(deptno=10,sal,0))AS"10",
		 SUM(if(deptno=20,sal,0))AS"20",
		 SUM(if(deptno=30,sal,0))AS"30"
	FROM emp;

mysql>
SELECT job,SUM(if(deptno=10,sal,0))AS"10",
		 SUM(if(deptno=20,sal,0))AS"20",
		 SUM(if(deptno=30,sal,0))AS"30"
	FROM emp
	GROUP BY job;

mysql>
SELECT job,count(if(deptno=10,sal,0))AS"10",
		 count(if(deptno=20,sal,0))AS"20",
		 count(if(deptno=30,sal,0))AS"30"
	FROM emp
	GROUP BY job;
```

```sql
mysql>
SELECT ename,IFNULL(comm,0)
FROM emp;
```

```sql
mysql>
SELECT DATE_FORMAT(HIREDATE,'%Y')
FROM emp;

mysql>
SELECT ename,hiredate
FROM emp
WHERE DATE_FORMAT(HIREDATE,'%w') = 3;

mysql>
SELECT ename,hiredate
FROM emp
WHERE DATE_FORMAT(hiredate,'%Y')=1981;

mysql>
SELECT ename,sal,hiredate
FROM emp
WHERE hiredate between str_to_date('1981/01/01','%Y/%m/%d')
AND STR_TO_DATE('1981/12/31','%Y/%m/%d')
and job = 'SALESMAN'
ORDER BY sal DESC;
```

※ Maria DB 의 주요날짜포멧

1. %Y : 연도 4자리
2. %y : 연도 2자리
3. %M : 달 (영문)
4. %m : 달 (숫자)
5. %W : 요일(영문)
6. %w : 요일(숫자 —> 0 : sunday, 1 : monday)
7. %d : 일(숫자)
8. %D : 일(th)

```sql
SELECT deptno,SUM(sal)
FROM emp
GROUP BY deptno WITH ROLLUP;

SELECT ifnull(job,'total') AS job2, SUM(sal)
FROM emp
GROUP BY job WITH ROLLUP;
```

```sql
SELECT deptno,GROUP_CONCAT(ename ORDER BY ename)
FROM emp
GROUP BY deptno;

SELECT concat(ename,sal)
FROM emp;

SELECT deptno,GROUP_CONCAT(CONCAT(ename,sal) ORDER BY ename)
FROM emp
GROUP BY deptno;

SELECT deptno,GROUP_CONCAT(CONCAT(ename,'(',sal,')') ORDER BY ename) AS grouping
FROM emp
GROUP BY deptno;
```

```sql
SELECT ename, sal, RANK() over(ORDER BY sal DESC) 'rank'
FROM emp;

SELECT ename, sal, RANK() over(ORDER BY sal DESC) 'rank'
FROM emp
LIMIT 1;

SELECT job,ename, sal, RANK() over(partition by job ORDER BY sal DESC) 'rank'
FROM emp
```

### **※ mariaDB(Mysql) auto commit 활성화(비활성화)**

```sql
SHOW VARIABLES LIKE 'autocommit%';

set autocommit = FALSE;
```

### ※ 선생님한테 물어보고싶은 쿼리

```sql
SELECT ifnull(job,'total'),ifnull(deptno,'-'), sum(sal)
FROM emp
GROUP BY job,deptno WITH ROLLUP
```

두번째 열의 총계부분과 소계부분을 다르게 적고싶습니다

union 사용하라고 하셨는데 잘 안돼서 다시 여쭤보고 싶습니다.

■ 숫자 수학적 표기법(천단위 콤마)

```python
oracle > select ename, to_char(sal,'999,999')
from emp;

mysql > SELECT ENAME, FORMAT(SAL,0)
FROM EMP;

mysql > SELECT ENAME, FORMAT(SAL*2000,0)
FROM EMP;

mysql > SELECT ENAME, FORMAT(SAL*2000,1)
FROM EMP;
```

```python
show tables;

create table emp2
( empno int(4),
	ename varchar(10),
	job varchar(10),
	mgr int(4),
	hiredate date,
	sal int(7),
	comm int(7),
  deptno int(4) );

CREATE TABLE dept2(
	deptno INT(10),
	dname VARCHAR(10),
	loc VARCHAR(10));
```

### ▥ 테이블에 csv 파일 로드하기

```python
:%s/,,/\\N,/g

%s/변경전문자/변경후문자,/g(모두)

,, 와 \\ 다 두개씩 씀
```

![Untitled](Linux%20%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A6%E1%84%89%E1%85%A5%20maria%20DB%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%2076cde0f6ce15401d8e8cce4c8c71d267/Untitled.png)

```python
load data local infile '/root/emp.csv'
replace
into table orcl.emp2
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 lines
(empno,ename,job,mgr,hiredate,sal,comm,deptno);
```

**※ 설명**

load data local infile '/root/emp.csv' ← 로드할 csv 파일명
replace ← 기존에 테이블에 데이터가 있다면 지금 로드할 데이터로 대체하겠다
into table orcl.emp2 ← orcl 데이터베이스의 emp2 테이블에 입력
fields terminated by ',' ← 값과 값은 콤마로 구분하겠다.
enclosed by '"' ← 데이터 중에 혹시 더블쿼테이션 마크로 둘러져있는 데이터도 입력되게
lines terminated by '\n' ← 행과 행은 엔터로 구분되어져 있다.
ignore 1 lines ← 첫번째 행은 무시하겠다.
(empno,ename,job,mgr,hiredate,sal,comm,deptno);

```python
truncate table emp2;

load data local infile '/root/emp.csv'
replace
into table orcl.emp2
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
(empno,ename,job,mgr,hiredate,sal,comm,deptno);

commit;
```

```bash
load data local infile '/root/dept.csv'
replace
into table orcl.dept2
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 lines
(deptno,dname,loc);
```

### ■ 네이버 영화 리뷰 데이터를 마리아 디비에 생성하기

```bash
load data local infile '/root/reviewData2.csv'
replace
into table orcl.naver2
fields terminated by ','
enclosed by '"'
lines terminated by '\r\n'
(cname,score,review);
```

```bash
select distinct(cname) from naver2;
```

### ■ virtual machine 에서 workbench 툴을 설치해서 maria db 사용

[https://cafe.daum.net/oracleoracle/Sho9/15](https://cafe.daum.net/oracleoracle/Sho9/15)

```bash
select review
from naver2
where review like '%조인성%';

select distinct(review)
from naver2
where review like '%최고%';

select cname,count(cname)
from naver2
where review like '%최고%'
group by cname;

select cname, count(score)
from naver2
where score = 1
group by cname
order by 2 desc limit 1;
```

### ▩ 내 자리에서 서버실에 있는 workbench 프로그램을 실행하는 방법

1.  mobaxterm → settings → configuration → x11 → x11 remote access → full

                                            → configuration → x11 → x11windowed mode: X11 server ~선택

2.  도스창을 열고 자신의 아이피 주소를 확인

```bash
ipconfig
```

1. 모바텀 터미널 창에서 다음과 같이 수행

```bash
export DISPLAY=자신의 ipv4 주소
export DISPLAY=192.168.0.6
mysql-workbench
```

### ▩ centos 에 아나콘다 설치하기

[https://cafe.daum.net/oracleoracle/Sho9/17](https://cafe.daum.net/oracleoracle/Sho9/17)
