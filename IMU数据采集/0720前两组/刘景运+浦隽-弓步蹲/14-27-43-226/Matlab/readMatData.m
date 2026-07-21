% 第1列(设备):1 表示 WT22222(f6:b1:93:b5:2b:23)   2 表示 WT901BLE11(f7:36:ca:b7:cb:34)   
% 第2列:片上时间   以开始时间作为起点，每条数据与开始时间的间隔（单位为秒）
% 第3列:加速度X(g)
% 第4列:加速度Y(g)
% 第5列:加速度Z(g)
% 第6列:角速度X(°/s)
% 第7列:角速度Y(°/s)
% 第8列:角速度Z(°/s)
% 第9列:位移X(mm)
% 第10列:位移Y(mm)
% 第11列:位移Z(mm)
% 第12列:速度X(mm/s)
% 第13列:速度Y(mm/s)
% 第14列:速度Z(mm/s)
% 第15列:角度X(°)
% 第16列:角度Y(°)
% 第17列:角度Z(°)
% 第18列:磁场X(ʯt)
% 第19列:磁场Y(ʯt)
% 第20列:磁场Z(ʯt)
% 第21列:温度(℃)
% 第22列:四元数0
% 第23列:四元数1
% 第24列:四元数2
% 第25列:四元数3
% 函数调用：a=readMatData;
function d = readMatData(file)

    if nargin<1
        disp('默认数据')
        file='data.mat';
    else
        disp(file);
    end

    disp('加载mat文件')
    load('data.mat')
    S=whos;
    len = length(S)-1;
    dend = eval(S(len).name);
    d1 = eval(S(1).name);
    len_m = length(d1);
    len_n = length(d1(1,:));

    d=zeros(len_m*(len-1)+length(dend),len_n);
    %h=waitbar(0,'数据合并中……');
    for i=1:len-1
        dTemp = eval(S(i).name);
        d(len_m*(i-1)+1:len_m*i,:)=[dTemp];
        m=len-1;
        %p=fix(i/(m)*len_m)/100; %这样做是可以让进度条的%位数为2位
        %str=['正在合并，目前进度为 ',num2str(p),' %，完成 ',num2str(i),'/',num2str(m)];%进度条上显示的内容
        %waitbar(i/m,h,str);
    end
    d(len_m*(len-1)+1:len_m*(len-1)+length(dend),:)=dend;

end