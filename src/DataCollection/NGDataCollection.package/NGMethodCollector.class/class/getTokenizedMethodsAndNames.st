as yet unclassified
getTokenizedMethodsAndNames

	| allClasses classes tokenizer methods |
	
	allClasses := Smalltalk globals allClasses.
	classes := (1 to: 2) collect: [ :i | allClasses at: i ].
	
	tokenizer := NGTokenizationPipeline new.
	tokenizer add: NGSyntacticTokenizer new.
	tokenizer add: NGWordTokenizer new.
	
	methods := classes flatCollect: [ :eachClass |
		eachClass methods collect: [ :eachMethod |
			{ tokenizer sourceAsTokenizedString: eachMethod . eachMethod selector asString } ] ].
	
	^ methods