tableR('as',A,'header',{'DoF','$\nu = 0.01$','$\nu = 0.1$','$\nu = 1$'},'format',{'%4.0f','%1.5f','%1.5f','%1.5f'})
%% Print Table
%
% This function will print an ASCII table, for example:
%
%              Decrease in Error, t_fo = 100
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%                mu=0.1        mu=0.5        mu=1.0      
%           .--------------------------------------------
%    h=0.01 | 8.692922e-01  8.530311e-01  4.018080e-01   
%    h=0.02 | 5.797046e-01  6.220551e-01  7.596669e-02   
%           |                                            
%    Factor | 1.449548e-01  5.132495e-01  1.233189e-01   
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
% Usage:
%
%       data = rand(2,3);% This could also be a cell.
%
%       table('This is the Title',data,...
%                 'header',{'Col-1','Col-2','Col-3'},...
%                 'rowheader',{'Row-1','Row-2'},...
%                 'border','|','format',{'%0.2f';'%0.3f'})
%
%       table('This is the Title',data)
%
%       table('Decrease in Error, t_fo = 100',rand(4,3),...
%             'header',{'mu=0.1','mu=0.5','mu=1.0'},...
%             'rowheader',{'h=0.01','h=0.02','','Factor'},...
%             'format',{'%e';'%e';'';'%e'})
%
%
%       data = rand(5,3);
%       for ii = 1:size(data,1);
%           % Do computations
%           table('This is the Title',data(1:ii,:),'printRow',true);
%           pause(0.5);
%       end
%
% 
% Options:                                          DEFAULTS
%
%       ALIGN:    {'left','right','center'}         'right'
%
%       DELIM:    STRING - goes between columns     '|'
%
%       PADDING:  STRING - either side of data      '  '
%
%       MARGIN:   STRING - outside of table         '  '
%
%       BORDER:   STRING - around table data        ''
%
%       FORMAT:   FPRINTF STRING - format of data   '%1.5e'
%                 Format can also be a cell array
%                 that is the size of data, or 
%                 only the size of cols or rows.
%
%                 e.g. {'%0.2f','%0.3f','%0.5f'}
%
%       HEADER:   CELL of STRINGS                   none
%
%       ROWHEADER: CELL of STRINGS                  none
%
%       PRINTROW:  BOOLEAN                          false
%                  This will print the last row only
%
%       FINALROW:  BOOLEAN                          false
%                  Indicates if this is the final row
%
%       LATEX:     BOOLEAN                          false
%                  This will output the source for
%                  a table in LaTEX
%
%
% Rowan Cockett
% 25-Oct-2012
% University of British Columbia
% rcockett@eos.ubc.ca

function tableR(title,data,varargin)

%-------------------- Begin Varargin Switch --------------------
% Set the default inputs
align = @right;
delim = '';
padding = ' ';
margin = '   ';
border = '';
format = {'%1.5e'};
useHeader = false;
useRowHeader = false;
printRow = false;
finalRow = false;
printLATEX = false;
showZeros = true;

if rem(numel(varargin),2) ~= 0
    error('VararginSwitch:InvalidNumberOfInputs','Incorrect number of input arguments')
end

% Parse the varargin
for pair=1:2:length(varargin)
    optName = varargin{pair};
    if ~ischar(optName)
        disp(optName);
        error('Invalid input. Must be a string.');
    end
    switch upper(optName)
        case 'ALIGN'
            align = varargin{pair+1};
            if ~ischar(align)
                error('Invalid form for ''align'' option pick {''right'',''left'',''center''}.')
            end
            switch upper(align)
                case 'RIGHT'
                    align = @right;
                case 'LEFT'
                    align = @left;
                case 'CENTER'
                    align = @center;
                otherwise
                    error('Invalid form for ''align'' option pick {''right'',''left'',''center''}.')
            end
        case 'DELIM'
            delim = varargin{pair+1};
            if ~ischar(delim)
                error('Invalid form for delim option. Delim must be a string.')
            end
        case 'PADDING'
            padding = varargin{pair+1};
            if ~ischar(padding)
                error('Invalid form for padding option. Padding must be a string.')
            end
        case 'MARGIN'
            margin = varargin{pair+1};
            if ~ischar(margin)
                error('Invalid form for margin option. Margin must be a string.')
            end
        case 'BORDER'
            border = varargin{pair+1};
            if ~ischar(border)
                error('Invalid form for border option. Border must be a string.')
            end
        case 'FORMAT'
            format = varargin{pair+1};
            if ischar(format)
                format = {format};
            elseif iscell(format)
                if all(size(format) == [1,size(data,2)])
                    format = repmat(format,size(data,1),1);
                elseif all(size(format) == [size(data,1),1])
                    format = repmat(format,1,size(data,2));
                elseif ~all(size(format) == size(data))
                    error('Format must be a string or a cell with the same size as data, columns, or rows.')
                end
            else
                error('Format must be a single string or a cell.')
            end
        case 'HEADER'
            if iscell(varargin{pair+1})
                if length(varargin{pair+1}) == size(data,2)
                    %These are column headers
                    useHeader = true;
                    columnHeaders = reshape(varargin{pair+1},1,size(data,2));
                else
                    error('Header must be the length of columns.')
                end
            else
                error('Header must be a cell vector.')
            end
        case 'ROWHEADER'
            if iscell(varargin{pair+1})
                if length(varargin{pair+1}) == size(data,1)
                    %These are column headers
                    useRowHeader = true;
                    rowHeaders = reshape(varargin{pair+1},size(data,1),1);
                else
                    error('RowHeader must be the length of rows.')
                end
            else
                error('RowHeader must be a cell vector.')
            end
        case 'PRINTROW'
            printRow = varargin{pair+1};
        case 'FINALROW'
            finalRow = varargin{pair+1};
        case 'LATEX'
            printLATEX = true;
        case 'SHOWZEROS'
            showZeros = varargin{pair+1};
        otherwise
            warning('VararginSwitch:NotRecognized',...
                ['The variable: ',optName,' was not recognized.']);
    end
end
%--------------------- End Varargin Switch ---------------------



strData = cell(size(data));

for ii = 1:numel(data)
    if iscell(data)
        d_ii = data{ii};
    else
        d_ii = data(ii);
    end
    if ischar(d_ii)
        strData{ii} = d_ii;
    else
        if ~showZeros && abs(d_ii) < 1E-11
            strData{ii} = '';
        else
            strData{ii} = sprintf(format{min(numel(format),ii)},d_ii);
        end
    end
end

% Fix column sizes
for ii = 1:size(data,2)
    mxSz = 0;
    if useHeader
        mxSz = length(columnHeaders{ii});
    end
    mxSz = max(mxSz,max(cellfun(@length,strData(:,ii))));
    strData(:,ii) = cellfun(@(str)(align(str,mxSz)),strData(:,ii),'uniformOutput',false);
    if useHeader
        columnHeaders{ii} = center(columnHeaders{ii},mxSz);
    end
end
        
rowHeaderFormat = '';
rowHeaderFormatTop = '';
if useRowHeader
    mxRSz = max(cellfun(@length,rowHeaders));
    rowHeaderFormat    = ['%s',padding,'|',padding];
    rowHeaderFormatTop = ['%s',padding,' ',padding];
    rowHeaders = cellfun(@(str)(left(str,mxRSz)),rowHeaders,'uniformOutput',false);
    strData = [rowHeaders,strData];
    if useHeader
        columnHeaders = [' ',columnHeaders];
        columnHeaders{1} = repmat(' ',mxRSz,1);
    end
end


% Print the table
line = [border,margin,rowHeaderFormat,repmat(['%s',padding,delim,padding],1,size(data,2)-1),'%s',margin,border,'\n'];
headerLine = [border,margin,rowHeaderFormatTop,repmat(['%s',padding,delim,padding],1,size(data,2)-1),'%s',margin,border,'\n'];
strData = strData';
n = length(sprintf(line,strData{:,1}))-1;
if ~printRow || size(strData,2) == 1
    fprintf('\n\n%s%s\n',repmat(' ',fix((n-length(title))/2),1),title)
    fprintf('%s\n',repmat('~',n,1));
    if useHeader
        fprintf(headerLine,columnHeaders{:});
        if useRowHeader
            nR = length(sprintf([margin,rowHeaderFormatTop],rowHeaders{1}));
            fprintf('%s%s.%s%s\n',border,repmat(' ',nR-2,1),repmat('-',n-nR+1-2*length(border),1),border);
        else
            fprintf('%s%s%s\n',border,repmat('-',n-2*length(border),1),border);
        end
    end

    fprintf(line,strData{:})
    if ~printRow
        fprintf('%s\n\n',repmat('~',n,1));
    end
else
    strData = strData(:,end);%only print the last row.
    fprintf(line,strData{:})
    if finalRow
        fprintf('%s\n\n',repmat('~',n,1));
    end
end
if isnumeric(data) && (~printRow || finalRow)
    % Print a link to copy out the data.
    str = sprintf(repmat('%e,',1,numel(data)),data(:));
    dataStr = ['tableData = reshape([',str(1:end-1),'],',num2str(size(data,1)),',',num2str(size(data,2)),')'];
    if useHeader
        str = sprintf(repmat('''%s'',',1,numel(columnHeaders)),columnHeaders{:});
        headerStr = ['{',str(1:end-1),'}'];
        disp(['<a href="matlab: ',dataStr,'">Data</a>, <a href="matlab: ',dataStr,'; table(''',title,''',tableData,''latex'',true,''header'',',headerStr,')">Latex</a>']);
        
    else
        disp(['<a href="matlab: ',dataStr,'">Data</a>, <a href="matlab: ',dataStr,'; table(''',title,''',tableData,''latex'',true)">Latex</a>']);
    end
end
if printLATEX
    rH_align = '';
    if useRowHeader;rH_align = 'l';end
    fprintf('\n\nCopy this code for a latex table:\n\n\\begin{table}[h]\n\\centering\n\\caption{%s}\n\\begin{tabular}{%s*{%i}{c}}\n\\hline\n',title,rH_align,size(strData,1));
    line = [repmat('%s    &    ',1,size(strData,1)-1),'%s  \\\\\n'];
    if useHeader
        fprintf(line,columnHeaders{:});
        fprintf('\\hline\n');
    end
    fprintf(line,strData{:});
    fprintf('\\hline\n\\end{tabular}\n\\label{tab:LABEL}\n\\end{table}\n\n')
end

function str = right(str,num)
str = [repmat(' ',1,num-length(str)),str];
function str = left(str,num)
str = [str,repmat(' ',1,num-length(str))];
function str = center(str,num)
str = [repmat(' ',1,ceil((num-length(str))/2)),str,repmat(' ',1,floor((num-length(str))/2))];