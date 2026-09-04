from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd

class TableError(Exception):
    pass

def _truth_array(x,index):
    if isinstance(x,pd.Series):
        return x.fillna(False).astype(bool)
    if isinstance(x,np.ndarray):
        if x.ndim!=1 or len(x)!=len(index):
            raise TableError("Boolean table expression has incompatible shape.")
        return pd.Series(x,index=index).fillna(False).astype(bool)
    return bool(x)

class _TableExpr(ast.NodeVisitor):
    """Safe row-wise expression evaluator over named dataframe columns."""

    def __init__(self,df,externals=None):
        self.df=df
        self.externals=dict(externals or {})

    def visit_Expression(self,n):
        return self.visit(n.body)

    def visit_Name(self,n):
        if n.id in self.df.columns:
            return self.df[n.id]
        if n.id in self.externals:
            value=self.externals[n.id]
            if np.asarray(value).ndim==0:
                return np.asarray(value).item()
        low=n.id.lower()
        if low=="true": return True
        if low=="false": return False
        raise TableError(f"Unknown column/scalar in FILTER: {n.id}")

    def visit_Constant(self,n):
        if isinstance(n.value,(int,float,bool,str)) or n.value is None:
            return n.value
        raise TableError("Unsupported FILTER constant.")

    def visit_UnaryOp(self,n):
        x=self.visit(n.operand)
        if isinstance(n.op,ast.Not):
            if isinstance(x,pd.Series):
                return ~x.fillna(False).astype(bool)
            return not bool(x)
        if isinstance(n.op,ast.USub): return -x
        if isinstance(n.op,ast.UAdd): return +x
        raise TableError("Unsupported unary operator in FILTER.")

    def visit_BinOp(self,n):
        a=self.visit(n.left); b=self.visit(n.right)
        if isinstance(n.op,ast.Add): return a+b
        if isinstance(n.op,ast.Sub): return a-b
        if isinstance(n.op,ast.Mult): return a*b
        if isinstance(n.op,ast.Div): return a/b
        if isinstance(n.op,ast.Mod): return a%b
        if isinstance(n.op,ast.Pow): return a**b
        raise TableError("Unsupported arithmetic operator in FILTER.")

    def visit_Compare(self,n):
        left=self.visit(n.left)
        out=None
        for op,comp in zip(n.ops,n.comparators):
            right=self.visit(comp)
            if isinstance(op,ast.Lt): cur=left<right
            elif isinstance(op,ast.LtE): cur=left<=right
            elif isinstance(op,ast.Gt): cur=left>right
            elif isinstance(op,ast.GtE): cur=left>=right
            elif isinstance(op,ast.Eq): cur=left==right
            elif isinstance(op,ast.NotEq): cur=left!=right
            else: raise TableError("Unsupported comparison in FILTER.")
            out=cur if out is None else (_truth_array(out,self.df.index) & _truth_array(cur,self.df.index))
            left=right
        return out

    def visit_BoolOp(self,n):
        vals=[_truth_array(self.visit(x),self.df.index) for x in n.values]
        if isinstance(n.op,ast.And):
            out=vals[0]
            for v in vals[1:]: out=out & v
            return out
        if isinstance(n.op,ast.Or):
            out=vals[0]
            for v in vals[1:]: out=out | v
            return out
        raise TableError("Unsupported boolean operator in FILTER.")

    def visit_Call(self,n):
        if not isinstance(n.func,ast.Name) or len(n.args)!=1 or n.keywords:
            raise TableError("FILTER functions must be one-argument direct calls.")
        name=n.func.id.upper()
        x=self.visit(n.args[0])
        if name=="MISSING":
            return pd.isna(x)
        if name=="PRESENT":
            return ~pd.isna(x)
        if name=="ABS":
            return np.abs(x)
        raise TableError(f"Unsupported FILTER function: {name}")

    def generic_visit(self,n):
        raise TableError(f"Unsupported FILTER expression element: {type(n).__name__}")

def _normalize_boolean_words(expr):
    # Python parser requires lowercase keywords.
    tokens=expr.strip()
    tokens=tokens.replace("&&"," and ").replace("||"," or ")
    import re
    tokens=re.sub(r"\bAND\b","and",tokens,flags=re.I)
    tokens=re.sub(r"\bOR\b","or",tokens,flags=re.I)
    tokens=re.sub(r"\bNOT\b","not",tokens,flags=re.I)
    return tokens

class TAYTable:
    def __init__(self,df):
        if isinstance(df,TAYTable):
            df=df.df
        if not isinstance(df,pd.DataFrame):
            raise TableError("TAYTable requires a pandas DataFrame.")
        self.df=df.reset_index(drop=True).copy()

    @classmethod
    def from_csv(cls,path):
        return cls(pd.read_csv(Path(path)))

    def copy(self):
        return TAYTable(self.df.copy())

    @property
    def columns(self):
        return list(self.df.columns)

    @property
    def shape(self):
        return self.df.shape

    def __len__(self):
        return len(self.df)

    def column(self,name):
        name=str(name)
        if name not in self.df.columns:
            raise TableError(f"Unknown column: {name}")
        return self.df[name].to_numpy(copy=True)

    def keep(self,columns):
        cols=[str(c).strip() for c in columns]
        missing=[c for c in cols if c not in self.df.columns]
        if missing:
            raise TableError(f"Unknown columns: {missing}")
        return TAYTable(self.df.loc[:,cols])

    def drop(self,columns):
        cols=[str(c).strip() for c in columns]
        missing=[c for c in cols if c not in self.df.columns]
        if missing:
            raise TableError(f"Unknown columns: {missing}")
        return TAYTable(self.df.drop(columns=cols))

    def filter(self,expr,externals=None):
        parsed=ast.parse(_normalize_boolean_words(expr),mode="eval")
        mask=_TableExpr(self.df,externals).visit(parsed)
        mask=_truth_array(mask,self.df.index)
        if isinstance(mask,bool):
            return self.copy() if mask else TAYTable(self.df.iloc[0:0])
        return TAYTable(self.df.loc[mask].reset_index(drop=True))

    def fill(self,column,strategy_or_value):
        col=str(column)
        if col not in self.df.columns:
            raise TableError(f"Unknown column: {col}")
        out=self.df.copy()
        strategy=str(strategy_or_value).upper() if isinstance(strategy_or_value,str) else None
        s=out[col]
        if strategy=="MEAN":
            value=s.mean()
        elif strategy=="MEDIAN":
            value=s.median()
        elif strategy=="MODE":
            modes=s.mode(dropna=True)
            if len(modes)==0:
                raise TableError(f"Cannot compute MODE for all-missing column: {col}")
            value=modes.iloc[0]
        elif strategy=="ZERO":
            value=0
        elif strategy=="DROP":
            return TAYTable(out.loc[~s.isna()].reset_index(drop=True))
        else:
            value=strategy_or_value
        out[col]=s.fillna(value)
        return TAYTable(out)

    def sort(self,column,descending=False):
        col=str(column)
        if col not in self.df.columns:
            raise TableError(f"Unknown column: {col}")
        return TAYTable(self.df.sort_values(col,ascending=not bool(descending),kind="mergesort").reset_index(drop=True))

    def group(self,by,specs):
        bycols=[str(x).strip() for x in by]
        for c in bycols:
            if c not in self.df.columns:
                raise TableError(f"Unknown group column: {c}")

        named={}
        allowed={"MEAN":"mean","MEDIAN":"median","SUM":"sum","MIN":"min","MAX":"max","COUNT":"count","STD":"std","NUNIQUE":"nunique"}
        for col,agg in specs:
            col=str(col).strip()
            agg=str(agg).upper()
            if col not in self.df.columns:
                raise TableError(f"Unknown summarized column: {col}")
            if agg not in allowed:
                raise TableError(f"Unsupported aggregation: {agg}")
            outname=f"{col}_{agg.lower()}"
            named[outname]=(col,allowed[agg])

        grouped=self.df.groupby(bycols,dropna=False,sort=True).agg(**named).reset_index()
        return TAYTable(grouped)

    def missing_count(self,column=None):
        if column is None:
            return int(self.df.isna().sum().sum())
        col=str(column)
        if col not in self.df.columns:
            raise TableError(f"Unknown column: {col}")
        return int(self.df[col].isna().sum())

    def to_csv(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.df.to_csv(path,index=False)

    def to_records(self):
        return self.df.to_dict(orient="records")

    def equals(self,other,check_dtype=False):
        if not isinstance(other,TAYTable):
            return False
        try:
            pd.testing.assert_frame_equal(
                self.df.reset_index(drop=True),
                other.df.reset_index(drop=True),
                check_dtype=check_dtype,
                check_like=False
            )
            return True
        except AssertionError:
            return False

    def __repr__(self):
        return f"TAYTable(rows={len(self.df)}, cols={len(self.df.columns)}, columns={self.columns})"
