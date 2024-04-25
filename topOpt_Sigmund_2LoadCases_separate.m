function top(nelx,nely,volfrac,penal,rmin);
% INITIALIZE
n_iter = 100;
x(1:nely,1:nelx) = volfrac; 
loop = 0; 
change = 1.;
% START ITERATION
while change > 0.01;
  loop = loop + 1;
  xold = x;

  %if loop > n_iter;
  %    break
  %end   
% FE-ANALYSIS
  [U]=FE(nelx,nely,x,penal);         
% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
  [KE] = lk;
  c = 0.;
% For two load two compliance objective case
  for ely = 1:nely
      for elx = 1:nelx
        n1 = (nely+1)*(elx-1)+ely;
        n2 = (nely+1)* elx + ely;
        dc(ely,elx) = 0.;
        for i = 1:2
            Ue = U([2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2], i);
            c = c + x(ely, elx)^penal*Ue'*KE*Ue;
            dc(ely,elx) = dc(ely,elx) - penal*x(ely,elx)^(penal-1)*Ue'*KE*Ue;
        end
      end
  end
% For single load case
  % for ely = 1:nely
  %   for elx = 1:nelx
  %     n1 = (nely+1)*(elx-1)+ely; 
  %     n2     = (nely+1)* elx   +ely;
  %     Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
  %     f_int = Ue'*KE*Ue;
  %     c = c + x(ely,elx)^penal*Ue'*KE*Ue;
  %     %disp(f_int);
  %     %disp(c);
  %     dc(ely,elx) = -penal*x(ely,elx)^(penal-1)*Ue'*KE*Ue;
  %   end
  % end
  %disp(sum(sum(KE)))
  %disp(sum(sum(Ue)))
  %disp(f_int)
  %disp(c)
% FILTERING OF SENSITIVITIES
  [dc]   = check(nelx,nely,rmin,x,dc);    
% DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
  [x]    = OC(nelx,nely,x,volfrac,dc); 
% PRINT RESULTS
  %disp(xold)
  %disp(x)
  %disp(max(max(abs(xold))));
  %disp(sum(sum(abs(xold))));
  change = max(max(abs(x-xold)));
  disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',c) ...
       ' Vol.: ' sprintf('%6.3f',sum(sum(x))/(nelx*nely)) ...
        ' ch.: ' sprintf('%6.3f',change )])
% PLOT DENSITIES  
  colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);
end 
%%%%%%%%%% OPTIMALITY CRITERIA UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xnew]=OC(nelx,nely,x,volfrac,dc)  
l1 = 0; l2 = 100000; move = 0.2;
while (l2-l1 > 1e-4)
  lmid = 0.5*(l2+l1);
  xnew = max(0.001,max(x-move,min(1.,min(x+move,x.*sqrt(-dc./lmid)))));
  if sum(sum(xnew)) - volfrac*nelx*nely > 0;
    l1 = lmid;
  else
    l2 = lmid;
  end
end
%%%%%%%%%% MESH-INDEPENDENCY FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dcn]=check(nelx,nely,rmin,x,dc)
dcn=zeros(nely,nelx);
for i = 1:nelx
  for j = 1:nely
    sum=0.0; 
    for k = max(i-floor(rmin),1):min(i+floor(rmin),nelx)
      for l = max(j-floor(rmin),1):min(j+floor(rmin),nely)
        fac = rmin-sqrt((i-k)^2+(j-l)^2);
        sum = sum+max(0,fac);
        dcn(j,i) = dcn(j,i) + max(0,fac)*x(l,k)*dc(l,k);
      end
    end
    dcn(j,i) = dcn(j,i)/(x(j,i)*sum);
  end
end
%%%%%%%%%% FE-ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U]=FE(nelx,nely,x,penal)
[KE] = lk; 
K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
%ONE LOAD CASE
% F = sparse(2*(nely+1)*(nelx+1),1); 
% U = zeros(2*(nely+1)*(nelx+1),1);
%TWO LOAD CASES
F = sparse(2*(nely+1)*(nelx+1),2);
U = sparse(2*(nely+1)*(nelx+1),2);
for elx = 1:nelx
  for ely = 1:nely
    n1 = (nely+1)*(elx-1)+ely;
    n2 = (nely+1)* elx   +ely;
    edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
    K(edof,edof) = K(edof,edof) + x(ely,elx)^penal*KE;
  end
end
%disp(K(end-3:end, end-3:end));
%disp(sum(sum(abs(KE))));
%disp(x(ely,elx));
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
% F(2,1) = -1;
%fixeddofs   = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);

%TWO LOADED PROBLEM
F(2*(nelx+1)*(nely+1),1) = -1.;
            F(2*(nelx)*(nely+1)+2,2) = 1.;

%DEFINE LOADS AND SUPPORTS (CANTILEVER)
%F(2*(nelx+1)*(nely+1),1) = -1;
fixeddofs = [1:2*(nely+1)];

alldofs     = [1:2*(nely+1)*(nelx+1)];
freedofs    = setdiff(alldofs,fixeddofs);
% SOLVING
U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
U(fixeddofs,:)= 0;
%%%%%%%%%% ELEMENT STIFFNESS MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE]=lk
E = 1.; 
nu = 0.3;
k=[ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
   -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];