close all
X = [0,2,6,10];
Y = [38100, 174000, 446000, 718000];

plot(X(1:2),Y(1:2), 'color', [0,0,1],"LineWidth",5);
grid on
hold on
plot(X(2:4),Y(2:4),'--','color', [0,0.8,0],"LineWidth",5);
plot(X(1:2),Y(1:2),'o','MarkerSize',10,'MarkerFaceColor',[1,0,0]);
text(X(1:2),Y(1:2),{'Initial Seed';'Current Status'},'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize',27)
yticks([0 200000 400000 600000])
xticks([0 2 4 6 8])
ax = gca;
ax.YAxis.Exponent=0;
xlabel(["","Months"])
ylabel(["Dataset Volume",""])
xlim([0 8])
ylim([0 600000])
title("WeatherStream Dataset")
set(gca,'fontsize',26)
pbaspect([2 2 1])
h=gcf;
set(h,'PaperPositionMode','auto');         
set(h,'PaperOrientation','portrait');
set(h,'Position',[50 50 1200 800]);
print(gcf, '-dsvg', 'test1.svg')