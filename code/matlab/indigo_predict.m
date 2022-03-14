function [test_interactions, testinteractions_scores, indigo_model, sigma_delta_scores]  = indigo_predict(indigo_model,testdata, input_type,annotation_filename,chemogenomics_filename,z,phenotype_data, phenotype_labels, conditions)
%[test_interactions, testinteractions_scores, indigo_model, sigma_delta_scores]  = indigo_predict(indigo_model,testdata, input_type,annotation_filename,chemogenomics_filename,z)
%%%% steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 load drug interaction data
% 2. convert  and match drug interaction data labels with chemogenomic data
% 3. input to indigo
%%%%%%%%%%% input processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if input_type == 1 % drug mode
    testdrugs = testdata;
elseif input_type == 2
    test_interactions1 = testdata;
else
    error('incorrect input: input_type is 1 (drug) or 2 (interaction)');
end

if ~exist('z','var') || isempty(z)
    z = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2a. match drugs with identifiers in chemogenomic data. 
%  convert  and match drug interaction data labels with chemical genetic data labels
 [num, txt] = xlsread(annotation_filename);
 [drugxn_id, chemgen_id] = deal(txt(:,1),txt(:,2));
 drugxn_id1 = sort(drugxn_id);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if input_type == 1
    drugnames_cell = testdrugs;
    for i = 1:length(drugxn_id)
        drugnames_cell (ismember(drugnames_cell,drugxn_id(i))) = chemgen_id(i);
    end
     drugxn_id1 = unique([drugxn_id;testdrugs]);
    alldrugcombinations = drugxn_id1(nchoosek(1:length(drugxn_id1),2));
    ix = ismember(alldrugcombinations, testdrugs); ix = any(ix,2);
    drugpairsname_cell = alldrugcombinations(ix,:);
     for i = 1:length(drugxn_id)
        drugpairsname_cell(ismember(drugpairsname_cell,drugxn_id(i))) = chemgen_id(i);
    end
    
elseif input_type == 2
    drugpairsname_cell = test_interactions1;
    for i = 1:length(drugxn_id)
        drugpairsname_cell(ismember(drugpairsname_cell,drugxn_id(i))) = chemgen_id(i);
    end
end
    testdrugs = unique(drugpairsname_cell(:));

%drug_abb_chem = drugs_all(~ismember(drugs_all,drugnames_cell)); %these drugs have chem gen data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2b. load and match with chemical genetic data
if ~exist('phenotype_data','var') || isempty(phenotype_data)
[phenotype_data, phenotype_labels, conditions] = process_chemgen(chemogenomics_filename,z);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ix = ismember(drugpairsname_cell,conditions); 
ix = (ix(:,1) & ix(:,2)); 
test_interactions = drugpairsname_cell(ix,:); %% these xns have chem gen data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. identify train and test drugs
[ix pos] = ismember(testdrugs,conditions); 
testchemgen = phenotype_data(:,pos);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4 input to indigo
[testinteractions_scores, indigo_model, sigma_delta_scores]  = indigo_rf([],[], [], [], testdrugs,testchemgen, test_interactions,2,indigo_model);
if length(testinteractions_scores) > 20 % standardize the data for large data sets
%     disp('outputting normalized interaction score')
% testinteractions_scores = zscore(testinteractions_scores);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end