(* ::Package:: *)

ClearAll["Global`*"];
SetDirectory[NotebookDirectory[]];
SetOptions[ListPlot,
  BaseStyle -> {FontFamily ->  "Gill Sans", FontSize -> 14}];
SetOptions[Manipulate,
  LabelStyle -> Directive[{FontFamily ->  "Gill Sans", FontSize -> 14}]];
ClearAll[data, fig1, fig2, col, hh, x, dataHashed];

data=Import["data_www.csv"]/.""-> Indeterminate;
col[x_]:=col[x]=Position[data[[1]], x,1][[1,1]];
countries = Union[data[[All, 1]]];

(* hashes the data into memory for fast access *)
With[{allColumns=data[[1]]}, 
	Table[
		Table[
			dataHashed[x[[col["country"]]],x[[col["year"]]], y]= x[[col[y]]],  
			{y, allColumns}],
		{x, Rest@data}]
];

(* selecting the poor countries *)
poorCountries = Cases[countries, x_/;dataHashed[x, 1970,"per_cap_dollar"]<10000];

(* prepare the data used for the figures *)
getDataToPlot[startYear_, endYear_, res_]:= Cases[
	Table[{ 
		{ 
			(* change in government foreign assets / Y *) 
			-N[100 ((dataHashed[c, endYear, "ppg_debt"] - 
				If[res, dataHashed[c, endYear, "reserves"],0.])
				/dataHashed[c, endYear, "dollar_gdp"]-(dataHashed[c, startYear, "ppg_debt"] - 
				If[res, dataHashed[c, startYear, "reserves"],0.])
				/dataHashed[c, startYear, "dollar_gdp"])
				/(endYear-startYear)],
			(* change in NFA / Y*)
			N[100((dataHashed[c, endYear, "assets"] - dataHashed[c, endYear, "liabilities"] )
				/dataHashed[c, endYear, "dollar_gdp"]-
				(dataHashed[c, startYear, "assets"] - dataHashed[c, startYear, "liabilities"])
				/dataHashed[c, startYear, "dollar_gdp"])/(endYear-startYear) ],
			(* change in GDPPC - 2%*) 
			N[100((dataHashed[c, endYear, "per_cap_gdp"]/
				dataHashed[c, startYear, "per_cap_gdp"])^(1/(endYear-startYear))
				-1 - 0.02)]
		},  c},  {c, poorCountries}],
	{{__Real},_}];

(* doing a figure *)
graph[data1_, conf_, xAxessLabel_]:= 
	Block[{x}, 
		With[{lm=LinearModelFit[data1[[All, 1]], {1, x}, x]},
			Column[{With[{blm= lm["SinglePredictionBands", ConfidenceLevel->conf]},
					Show[ArrayReshape[#, {3}, "Periodic"]&@{
						ListPlot[Apply[Tooltip,data1, 1], Frame->True, PlotMarkers->Automatic], 
						Plot[Evaluate@{lm[x], blm}, {x, -10, 10},  
							Filling->{2->{1}, 3->{1}},
							PlotStyle->{{Thick, Black}, {Thin,Blue}, {Thin,Blue}} ]
						},
						ImageSize-> 500, 
						AxesLabel->{xAxessLabel, "Change in GDPPC relative to US"}
					]
				]
				, "\n"
				"{Intercept, Slope}: "<>ToString@lm["BestFitParameters"], 
				"2 \[Times] Standard Errors: "<>ToString@(2 lm["ParameterErrors"])
				}
			]
		]
	];

(* creating all three figures *)
fig[startYear_,endYear_, res_, conf_]:= Module[{dataToPlot= getDataToPlot[startYear, endYear, res]},
		Column[{
			Framed[graph[{{#[[1, 1]], #[[1, 3]]}, #[[2]]}&/@ dataToPlot, conf, "\!\(\*FractionBox[\(\[CapitalDelta] \((\*SubscriptBox[\(Gov\), \(Foreign\\\ Assets\)]/Y)\)\), \(T\)]\)"]], 
			Framed[graph[{{#[[1, 2]], #[[1, 3]]}, #[[2]]}&/@ dataToPlot, conf, "\!\(\*FractionBox[\(\[CapitalDelta] \((NFA/Y)\)\), \(T\)]\)"]], 
			Framed@graph[{{#[[1, 2]]- #[[1, 1]], #[[1, 3]]}, #[[2]]}&/@ dataToPlot, conf, "\!\(\*FractionBox[\(\[CapitalDelta] \((\*SubscriptBox[\(Priv\), \(Foreign\\\ Assets\)]/Y)\)\), \(T\)]\)"]
		}]
	];

fig[1970, 2004, True,0.6827]
