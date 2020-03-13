% Implementation of four different schemes applied to the Zakharov-Kuznetsov
% equation, as is is given in Eidnes and Li: "Linearly implicit local and
% global energy-preserving methods for Hamiltonian PDEs" [EL19].
% The schemes are:
% ** the LILEP and LIGEP schemes of [EL19],
% ** two-dimensional extensions of the LEP and GEP schemes of
% Gong, Cai and Wang: "Some new structure-preserving algorithms for general
% multi-symplectic formulations of Hamiltonian PDEs" (2014)
% 
% Code written by Sølve Eidnes and Lu Li

P = 30; % Length and width of the square spatial domain
M = 45; % Number of spatial steps in each direction
MM = M*M; % Number of spatial discretization points
h = P/M; % Spatial step size in each direction
T = 15; % End time
k = .1; % Time step size
N = round(T/k); % Number of time steps

% Initial conditions:
x = (0:h:P-h)';
y = (0:h:P-h)';
c = 2;
u0 = 3*c*sech(.5*sqrt(c)*(x-P/2)).^2 + .1*randn(1,length(y)); % Test problem in [EL19]
u0 = reshape(u0,[MM,1]);

% Creating the average and forward difference matrices:
e = ones(MM,1);
em = zeros(MM,1); em(1:M:MM) = 1;
ep = e; ep(M+1:M:MM) = 0;
Ax = .5*spdiags([em, e, ep],[-M+1,0,1],MM,MM);
Ay = .5*spdiags([e, e],[0,M],MM,MM);
Ay(MM-M+1:MM,1:M) = .5*speye(M);
deltax = 1/h*spdiags([em, -e, ep],[-M+1,0,1],MM,MM);
deltay = 1/h*spdiags([-e, e],[0,M],MM,MM);
deltay(MM-M+1:MM,1:M) = 1/h*speye(M);
I = speye(MM);

uNs = zeros(4,M,M); % To store solution values from all four methods
% To store discrete approximations to the energy at every time step:
energies = zeros(1,N);
energiesK = zeros(1,N);

% LILEP
u = u0;
tic
if mod(M,2) == 0 % if M even, in which case the matrix A below will be singular
    options = optimoptions(@fsolve,'Display','off','Algorithm','levenberg-marquardt',...
        'SpecifyObjectiveGradient',true,'PrecondBandWidth',0,'TolFun',1e-10);
    for i = 1:N
        f = @(un) fandjac_ZK_lilep(u,un,deltax,deltay,Ax,Ay,k,MM);
        un = fsolve(f,u,options);
        energies(i) = h^2*1/6*sum(3*((deltay*Ax*un).^2+(deltax*Ay*un).^2) - (Ax*Ay*un).^3);
        energiesK(i) = h^2*1/6*sum(2*(deltay*Ax*un).*(deltay*Ax*u)+(deltay*Ax*u).^2+2*(deltax*Ay*u).*(deltax*Ay*un)+(deltax*Ay*u).^2 - ((Ax*Ay*u).^2).*(Ax*Ay*un));
        u = un;
    end
else % if M odd
    for i = 1:N
        A = 1/k*Ax^3*Ay^2 + .5*deltax^3*Ay^2 + .5*deltax*deltay^2*Ax^2 + .5*deltax*Ax*Ay*spdiags(Ax*Ay*u,0,MM,MM)*Ax*Ay; % This matrix is singular if M is even
        B = 1/k*Ax^3*Ay^2*u - .5*deltax^3*Ay^2*u - .5*deltax*deltay^2*Ax^2*u;
        un = A\B;
        energies(i) = h^2*1/6*sum(3*((deltay*Ax*un).^2+(deltax*Ay*un).^2) - (Ax*Ay*un).^3);
        energiesK(i) = h^2*1/6*sum(2*(deltay*Ax*un).*(deltay*Ax*u)+(deltay*Ax*u).^2+2*(deltax*Ay*u).*(deltax*Ay*un)+(deltax*Ay*u).^2 - ((Ax*Ay*u).^2).*(Ax*Ay*un));
        u = un;
    end
end
fprintf('The LILEP scheme used %f seconds.\n',toc);
u = reshape(u,M,M)';
uNs(1,:,:) = u;
figure(2)
surf(x,y,u,'EdgeColor','none')
set(gca,'fontsize',9)
title('LILEP')
ylabel('y')
xlabel('x')
zlabel('u(x,y,t)')

% LEP
u = u0;
tic
if mod(M,2) == 0 % if M even
    for i = 1:N
        f = @(un) fandjac_ZK_lep(u,un,deltax,deltay,Ax,Ay,k,MM);
        un = fsolve(f,u,options);
        energies(i) = h^2*1/6*sum(3*((deltay*Ax*un).^2+(deltax*Ay*un).^2) - (Ax*Ay*un).^3);
        energiesK(i) = h^2*1/6*sum(2*(deltay*Ax*un).*(deltay*Ax*u)+(deltay*Ax*u).^2+2*(deltax*Ay*u).*(deltax*Ay*un)+(deltax*Ay*u).^2 - ((Ax*Ay*u).^2).*(Ax*Ay*un));
        u = un;
    end
else % if M odd
    for i = 1:N
        un = newton_ZK_LEP(u,u,deltax,deltay,Ax,Ay,k,MM,1e-10);
        energies(i) = h^2*1/6*sum(3*((deltay*Ax*un).^2+(deltax*Ay*un).^2) - (Ax*Ay*un).^3);
        energiesK(i) = h^2*1/6*sum(2*(deltay*Ax*un).*(deltay*Ax*u)+(deltay*Ax*u).^2+2*(deltax*Ay*u).*(deltax*Ay*un)+(deltax*Ay*u).^2 - ((Ax*Ay*u).^2).*(Ax*Ay*un));
        u = un;
    end
end
fprintf('The LEP scheme used %f seconds.\n',toc);
u = reshape(u,M,M)';
uNs(2,:,:) = u;
figure(3)
surf(x,y,u,'EdgeColor','none')
set(gca,'fontsize',9)
title('LEP')
ylabel('y')
xlabel('x')
zlabel('u(x,y,t)')

% Creating the central difference matrices used in LIGEP and GEP:
emm = zeros(MM,1); emm(M:M:MM) = 1;
ep = e; ep(M:M:MM) = 0;
epp = e; epp(M+1:M:MM) = 0;
deltacx = .5/h*spdiags([em, -ep, epp, -emm],[-M+1,-1,1,M-1],MM,MM);
deltacy = .5/h*spdiags([-e, e],[-M,M],MM,MM);
deltacy(MM-M+1:MM,1:M) = .5/h*speye(M);
deltacy(1:M,MM-M+1:MM) = -.5/h*speye(M);
D1x = deltacx;
D1y = deltacy;

% LIGEP
u = u0;
tic
for i = 1:N
    A = I/k + .5*D1x^3 + .5*D1x*D1y^2 + .5*D1x*spdiags(u,0,MM,MM);
    B = 1/k*u - .5*D1x^3*u - .5*D1x*D1y^2*u;
    un = A\B;
    energies(i) = h^2*1/6*sum(3*((D1y*un).^2+(D1x*un).^2) - un.^3);
    energiesK(i) = h^2*1/6*sum(2*(D1y*un).*(D1y*u)+(D1y*u).^2+2*(D1x*u).*(D1x*un)+(D1x*u).^2 - (u.^2.*un));
    u = un;
end
fprintf('The LIGEP scheme used %f seconds.\n',toc);
u = reshape(u,M,M)';
uNs(3,:,:) = u;
figure(4)
surf(x,y,u,'EdgeColor','none')
set(gca,'fontsize',9)
title('LIGEP')
ylabel('y')
xlabel('x')
zlabel('u(x,y,t)')

% GEP
u = u0;
tic
for i = 1:N
    un = newton_ZK_GEP(u,u,D1x,D1y,k,MM,I,1e-10);
    energies(i) = h^2*1/6*sum(3*((D1y*un).^2+(D1x*un).^2) - un.^3);
    energiesK(i) = h^2*1/6*sum(2*(D1y*un).*(D1y*u)+(D1y*u).^2+2*(D1x*u).*(D1x*un)+(D1x*u).^2 - (u.^2.*un));
    u = un;
end
fprintf('The GEP scheme used %f seconds.\n',toc);
u = reshape(u,M,M)';
uNs(4,:,:) = u;
figure(5)
surf(x,y,u,'EdgeColor','none')
set(gca,'fontsize',9)
title('GEP')
ylabel('y')
xlabel('x')
zlabel('u(x,y,t)')

%
function [f,J] = fandjac_ZK_lilep(u,un,deltax,deltay,Ax,Ay,k,MM)
f = 1/k*Ax^3*Ay^2*(un-u) + .5*deltax^3*Ay^2*(un+u) +...
    .5*deltax*deltay^2*Ax^2*(un+u) + .5*deltax*Ax*Ay*((Ax*Ay*u).*(Ax*Ay*un));
J = 1/k*Ax^3*Ay^2 + .5*deltax^3*Ay^2 + .5*deltax*deltay^2*Ax^2 +...
    .5*deltax*Ax*Ay*spdiags(Ax*Ay*u,0,MM,MM)*Ax*Ay;
end

function [f,J] = fandjac_ZK_lep(u,un,deltax,deltay,Ax,Ay,k,MM)
f = 1/k*Ax^3*Ay^2*(un-u) + .5*deltax^3*Ay^2*(un+u) + .5*deltax*deltay^2*Ax^2*(un+u) +...
        1/6*deltax*Ax*Ay*((Ax*Ay*un).^2 + (Ax*Ay*un).*(Ax*Ay*u) + (Ax*Ay*u).^2);
J = 1/k*Ax^3*Ay^2 + .5*deltax^3*Ay^2 + .5*deltax*deltay^2*Ax^2 +...
    1/6*deltax*Ax*Ay*(2*spdiags(Ax*Ay*un,0,MM,MM)*Ax*Ay+spdiags(Ax*Ay*u,0,MM,MM)*Ax*Ay);
end

function un = newton_ZK_LEP(u,un,deltax,deltay,Ax,Ay,k,MM,tol)
f = @(un) 1/k*Ax^3*Ay^2*(un-u) + .5*deltax^3*Ay^2*(un+u) + .5*deltax*deltay^2*Ax^2*(un+u) +...
        1/6*deltax*Ax*Ay*((Ax*Ay*un).^2 + (Ax*Ay*un).*(Ax*Ay*u) + (Ax*Ay*u).^2);
J = @(un) 1/k*Ax^3*Ay^2 + .5*deltax^3*Ay^2 + .5*deltax*deltay^2*Ax^2 +...
    1/6*deltax*Ax*Ay*(2*spdiags(Ax*Ay*un,0,MM,MM)*Ax*Ay+spdiags(Ax*Ay*u,0,MM,MM)*Ax*Ay);
err = norm(f(un));
k = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    k = k+1;
    if k > 5
        break;
        err
    end
end
end

function un = newton_ZK_GEP(u,un,Dx,Dy,k,MM,I,tol)
f = @(un) 1/k*(un-u) + .5*Dx^3*(un+u) + .5*Dx*Dy^2*(un+u) + 1/6*Dx*(un.^2 + un.*u + u.^2);
J = @(un) 1/k*I + .5*Dx^3 + .5*Dx*Dy^2 + 1/6*Dx*(2*spdiags(un,0,MM,MM)+spdiags(u,0,MM,MM));
err = norm(f(un));
k = 0;
while err > tol
    un = un - J(un)\f(un);
    err = norm(f(un));
    k = k+1;
    if k > 5
        break;
        err
    end
end
end