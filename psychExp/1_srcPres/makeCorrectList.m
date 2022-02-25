function correctList=makeCorrectList(N,exp,possible_cor,special_act)
  
    %correctList fuer Phasen

    % cluster, blocklength
    N_temp=N;
    act1List=ones(1,floor(N_temp/3))*possible_cor(1);
    N_temp=N_temp-floor(N_temp/3);
    act2List=ones(1,floor(N_temp/2))*possible_cor(2);
    N_temp=N_temp-floor(N_temp/2);
    act3List=ones(1,ceil(N_temp))*possible_cor(3);
    correctList=[act1List, act2List, act3List];
    again=1;
    while again>0
        correctList=correctList(randperm(length(correctList)));
        again=sum(abs(diff(correctList))==0);
    end
    
    % frequency
    others=possible_cor(possible_cor~=special_act);
    othersList=[ones(1,floor(N/4))*others(1),ones(1,ceil(N/4))*others(2)];
    if exp==2
        for idx=[1:N]
            if mod(idx,2)==0
                correctList(idx)=special_act;
            else
                randIdx=randi(length(othersList));
                correctList(idx)=othersList(randIdx);
                othersList(randIdx)=[];
            end
        end
    end
    
    % all correct
    if exp==4
        acts=randperm(5);
        while length(acts)<N
            next=randperm(5);
            while acts(end)==next(1), next=randperm(5); end
            acts=[acts,next];
        end
        correctList=acts(1:N);
    end

    
    fprintf('correctList = %d.\n',correctList);
    
    
end



