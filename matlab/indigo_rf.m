function [testinteractions_scores, indigo_model, sigma_delta_scores]  = indigo_rf(traindrugs,trainchemgen, train_interactions, trainxnscores, testdrugs,testchemgen, test_interactions,indigo_mode,indigo_model)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check if it is training mode (1) or testing mode (2)
% calculate sigma and delta scores from chemogenomic data
if (indigo_mode == 1) || (indigo_mode == 0)
    chemgen = trainchemgen;
    alldrugs = traindrugs;
    interactions = train_interactions;
    
    for i = 1:size(interactions,1)
        ix1 = ismember(alldrugs, interactions(i,1));
        ix2 = ismember(alldrugs, interactions(i,2));
        t1 = chemgen(:,ix1); t2 = chemgen(:,ix2);
        Xl = [t1,t2];
        traindiffdat1xxz2(:,i) = [t1 + t2;[(sum(logical(Xl')) ==1)]'];
        
    end
    
    
elseif (indigo_mode == 2) || (indigo_mode == 0) % testing mode
    chemgen = testchemgen;
    alldrugs = testdrugs;
    interactions = test_interactions;
    
    for i = 1:size(interactions,1)
        ix1 = ismember(alldrugs, interactions(i,1));
        ix2 = ismember(alldrugs, interactions(i,2));
        t1 = chemgen(:,ix1); t2 = chemgen(:,ix2);
        Xl = [t1,t2];
        testdiffdat1xxz2(:,i) = [t1 + t2;[(sum(logical(Xl')) ==1)]'];
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% run RF
if (indigo_mode == 1) || (indigo_mode == 0)
    indigo_model = fitrensemble(traindiffdat1xxz2',trainxnscores,'Method','Bag');
    sigma_delta_scores = traindiffdat1xxz2;
    testinteractions_scores = [];
elseif (indigo_mode == 2) || (indigo_mode == 0)
    testinteractions_scores = predict(indigo_model,testdiffdat1xxz2');
    sigma_delta_scores = testdiffdat1xxz2;
end


end